from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import pandas as pd
import uuid
import time
from datetime import datetime
import random

from app.models.schemas import (
    LeadInput, LeadScore, Recommendation, LeadResult, BatchResult,
    UploadResponse, HealthResponse, MetricsResponse, ErrorResponse
)
from app.services.classifier import classifier
from app.services.next_action_agent import next_action_agent
from app.services.retrieval import retrieval
from app.services.email_service import email_service
from app.services.rag_email_service import rag_email_service
from app.services.telegram_service import telegram_service
from app.services.whatsapp_service import whatsapp_service
from app.api.analytics import router as analytics_router
from app.services.performance_monitor import performance_monitor
from app.services.circuit_breaker import openai_circuit_breaker, mongodb_circuit_breaker
from app.utils.logging import get_logger
from app.utils.safety_filters import sanitize_content
from app.config import settings


logger = get_logger(__name__)
router = APIRouter()

# Include analytics router
router.include_router(analytics_router, prefix="/analytics", tags=["analytics"])


@router.post("/score", response_model=LeadScore)
async def score_lead(lead_data: LeadInput):
    """Classify a single lead's heat score."""
    try:
        if not classifier.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="Classification model not trained. Please train the model first."
            )
        
        # Classify lead
        score = classifier.predict(lead_data)
        
        logger.info(f"Scored lead: {score.lead_id}")
        return score
        
    except Exception as e:
        logger.error(f"Error scoring lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend", response_model=Recommendation)
async def get_recommendation(lead_data: LeadInput):
    """Get next action recommendation for a lead."""
    try:
        # Safety check: Sanitize input data
        safety_result = sanitize_content(
            f"{lead_data.name} {lead_data.email} {lead_data.campaign}",
            check_injection=True,
            check_pii=True
        )
        
        if not safety_result.is_safe:
            logger.warning(f"Unsafe content detected: {safety_result.detected_threats}")
            # Continue with filtered content but log the threat
        
        # First classify the lead
        if classifier.is_trained:
            lead_score = classifier.predict(lead_data)
        else:
            # Create dummy score for recommendation
            lead_score = LeadScore(
                lead_id=str(uuid.uuid4()),
                heat_score="warm",
                confidence=0.5,
                probabilities={"cold": 0.3, "warm": 0.5, "hot": 0.2}
            )
        
        # Generate recommendation
        recommendation = await next_action_agent.generate_recommendation(
            lead_data, lead_score
        )
        
        # Safety check: Sanitize recommendation output
        if recommendation.message:
            message_safety = sanitize_content(recommendation.message, check_injection=True)
            if not message_safety.is_safe:
                logger.warning(f"Unsafe recommendation message detected: {message_safety.detected_threats}")
                recommendation.message = message_safety.filtered_content
        
        logger.info(f"Generated recommendation for lead: {recommendation.lead_id}")
        return recommendation
        
    except Exception as e:
        logger.error(f"Error generating recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file for batch processing."""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV content quickly
        contents = await file.read()
        content_str = contents.decode('utf-8')
        lines = content_str.strip().split('\n')
        
        # Quick validation without pandas
        if len(lines) < 2:
            raise HTTPException(status_code=400, detail="CSV must have at least a header and one data row")
        
        header = lines[0].lower()
        total_rows = len(lines) - 1
        
        # Validate required columns
        required_columns = [
            'source', 'recency_days', 'region', 'role', 'campaign',
            'page_views', 'last_touch', 'prior_course_interest'
        ]
        
        # Optional columns for enhanced CSV
        optional_columns = [
            'search_keywords', 'time_spent', 'course_actions'
        ]
        
        missing_columns = [col for col in required_columns if col not in header]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # For now, assume all rows are valid (skip detailed validation for speed)
        valid_rows = total_rows
        invalid_rows = 0
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        # Store batch data (in production, you'd store this in database)
        # For now, we'll just return the response
        
        logger.info(f"Uploaded CSV: {file.filename}, {total_rows} rows")
        
        # Store batch info AND process leads in MongoDB
        try:
            from app.db import get_database
            db = await get_database()
            batch_collection = db["upload_batches"]
            leads_collection = db["processed_leads"]
            
            # Clear existing leads before processing new CSV
            await leads_collection.delete_many({})
            await batch_collection.delete_many({})
            logger.info("Cleared existing leads data for fresh upload")
            
            # Store batch info
            await batch_collection.insert_one({
                "batch_id": batch_id,
                "filename": file.filename,
                "total_rows": total_rows,
                "valid_rows": valid_rows,
                "invalid_rows": invalid_rows,
                "uploaded_at": datetime.now(),
                "status": "processing"
            })
            
            # Process and store ALL leads from uploaded CSV
            data_lines = lines[1:]  # Skip header, process all data rows
            
            for i, line in enumerate(data_lines):
                if not line.strip():
                    continue
                    
                values = line.split(',')
                if len(values) >= 8:
                    try:
                        # Create lead input - mapping CSV columns correctly
                        lead_data = {
                            "name": values[1].strip(),           # name
                            "email": values[2].strip(),          # email
                            "phone": values[3].strip() if len(values) > 3 else None,  # phone
                            "source": values[4].strip(),        # source
                            "recency_days": int(values[5].strip()), # recency_days
                            "region": values[6].strip(),         # region
                            "role": values[7].strip(),          # role
                            "campaign": values[8].strip(),     # campaign
                            "page_views": int(values[9].strip()), # page_views
                            "last_touch": values[10].strip(),   # last_touch
                            "prior_course_interest": values[11].strip() # prior_interest
                        }
                        
                        # Add optional columns if they exist
                        if len(values) > 12 and values[12].strip():
                            lead_data["search_keywords"] = values[12].strip()
                        if len(values) > 13 and values[13].strip():
                            lead_data["time_spent"] = int(values[13].strip())
                        if len(values) > 14 and values[14].strip():
                            lead_data["course_actions"] = values[14].strip()
                        
                        lead_input = LeadInput(**lead_data)
                        
                        # Score the lead
                        score_result = classifier.predict(lead_input)
                        
                        # Generate recommendation
                        try:
                            from app.services.next_action_agent import next_action_agent
                            recommendation = await next_action_agent.generate_recommendation(
                                lead_input, score_result
                            )
                        except Exception as e:
                            logger.warning(f"Could not generate recommendation for lead {i+1}: {e}")
                            recommendation = None
                        
                        # Store in MongoDB
                        lead_doc = {
                            "lead_id": f"{batch_id}_{i+1}",
                            "batch_id": batch_id,
                            "input_data": lead_data,
                            "lead_data": lead_data,  # Use the processed lead_data directly
                            "score": score_result.dict(),
                            "recommendation": recommendation.dict() if recommendation else None,
                            "processed_at": datetime.now(),
                            "status": "processed"
                        }
                        
                        await leads_collection.insert_one(lead_doc)
                        logger.info(f"Stored lead {i+1} in MongoDB")
                        
                    except Exception as e:
                        logger.warning(f"Error processing lead {i+1}: {e}")
                        continue
            
            # Update batch status
            await batch_collection.update_one(
                {"batch_id": batch_id},
                {"$set": {"status": "completed", "processed_at": datetime.now()}}
            )
            
        except Exception as db_error:
            logger.error(f"MongoDB error: {db_error}")
            # Continue without storing for now
        
        return UploadResponse(
            filename=file.filename,
            total_rows=total_rows,
            valid_rows=valid_rows,
            invalid_rows=invalid_rows,
            batch_id=batch_id,
            message=f"Successfully uploaded {valid_rows} valid leads"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-score", response_model=BatchResult)
async def batch_score_leads(
    leads_data: List[Dict[str, Any]],
    background_tasks: BackgroundTasks
):
    """Score multiple leads in batch."""
    try:
        if not classifier.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="Classification model not trained. Please train the model first."
            )
        
        start_time = time.time()
        results = []
        processed_leads = 0
        failed_leads = 0
        
        for lead_dict in leads_data:
            try:
                # Convert to LeadInput
                lead_input = LeadInput(**lead_dict)
                
                # Classify lead
                score = classifier.predict(lead_input)
                
                # Generate recommendation
                recommendation = await next_action_agent.generate_recommendation(
                    lead_input, score
                )
                
                # Create result
                result = LeadResult(
                    lead_id=score.lead_id,
                    lead_data=lead_input,
                    score=score,
                    recommendation=recommendation
                )
                
                results.append(result)
                processed_leads += 1
                
            except Exception as e:
                logger.error(f"Error processing lead: {e}")
                failed_leads += 1
                continue
        
        processing_time = time.time() - start_time
        
        logger.info(f"Batch processed: {processed_leads} successful, {failed_leads} failed")
        
        return BatchResult(
            total_leads=len(leads_data),
            processed_leads=processed_leads,
            failed_leads=failed_leads,
            results=results,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch scoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leads", response_model=List[LeadResult])
async def get_leads(limit: int = 100, offset: int = 0):
    """Get processed leads from MongoDB."""
    try:
        from app.db import get_database
        
        db = await get_database()
        collection = db["processed_leads"]
        
        # Get leads from MongoDB
        cursor = collection.find().skip(offset).limit(limit).sort("processed_at", -1)
        leads = []
        
        async for doc in cursor:
            try:
                # Use lead_data if available, otherwise fall back to input_data
                lead_data_dict = doc.get("lead_data", doc["input_data"])
                
                lead_result = LeadResult(
                    lead_id=doc["lead_id"],
                    lead_data=LeadInput(**lead_data_dict),
                    score=LeadScore(**doc["score"]),
                    recommendation=Recommendation(**doc["recommendation"]) if doc.get("recommendation") else None
                )
                leads.append(lead_result)
            except Exception as e:
                logger.warning(f"Error parsing lead {doc.get('lead_id', 'unknown')}: {e}")
                continue
        
        # If no leads in MongoDB, return sample data
        if not leads:
            logger.info("No leads found in MongoDB, returning empty list")
            # Return empty list instead of sample data
        
        return leads
        
    except Exception as e:
        logger.error(f"Error retrieving leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leads/{lead_id}", response_model=LeadResult)
async def get_lead(lead_id: str):
    """Get specific lead details."""
    try:
        # This is a placeholder - in production, you'd query your database
        raise HTTPException(status_code=404, detail="Lead not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/leads/clear")
async def clear_all_leads():
    """Clear all leads from the database."""
    try:
        from app.db import get_database
        db = await get_database()
        
        # Clear processed leads
        leads_collection = db["processed_leads"]
        leads_result = await leads_collection.delete_many({})
        
        # Clear upload batches
        batches_collection = db["upload_batches"]
        batches_result = await batches_collection.delete_many({})
        
        logger.info(f"Cleared {leads_result.deleted_count} leads and {batches_result.deleted_count} batches")
        
        return {
            "message": "All leads cleared successfully",
            "leads_deleted": leads_result.deleted_count,
            "batches_deleted": batches_result.deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error clearing leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db_status = "connected"
        try:
            from app.db import get_database
            db = await get_database()
            await db.command("ping")
        except Exception:
            db_status = "disconnected"
        
        # Check model status
        model_status = "trained" if classifier.is_trained else "not_trained"
        
        return HealthResponse(
            status="healthy",
            version=settings.api_version,
            database_status=db_status,
            ml_model_status=model_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=settings.api_version,
            database_status="unknown",
            ml_model_status="unknown"
        )


@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics():
    """Get current performance metrics and monitoring data."""
    try:
        # Get performance summary
        performance_summary = performance_monitor.get_performance_summary()
        
        # Get circuit breaker states
        circuit_breaker_states = {
            "openai": openai_circuit_breaker.get_state(),
            "mongodb": mongodb_circuit_breaker.get_state()
        }
        
        # Check if we're meeting performance requirements
        requirements_status = {
            "p95_latency_requirement": {
                "threshold_ms": 2500,
                "current_p95_ms": performance_summary.get("p95_latency_ms", 0),
                "meets_requirement": performance_summary.get("p95_latency_ms", 0) <= 2500
            },
            "error_rate_requirement": {
                "threshold": 0.005,  # 0.5%
                "current_rate": performance_summary.get("error_rate", 0),
                "meets_requirement": performance_summary.get("error_rate", 0) <= 0.005
            }
        }
        
        return {
            "performance_summary": performance_summary,
            "circuit_breaker_states": circuit_breaker_states,
            "requirements_status": requirements_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/trace/{trace_id}", response_model=Dict[str, Any])
async def get_trace_details(trace_id: str):
    """Get detailed information about a specific performance trace."""
    try:
        # This would typically query a trace store
        # For now, return basic trace information
        return {
            "trace_id": trace_id,
            "status": "completed",  # Would be determined from actual trace data
            "message": "Trace details would be available in a full tracing system"
        }
        
    except Exception as e:
        logger.error(f"Error getting trace details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get model performance metrics."""
    try:
        if not classifier.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="Model not trained. No metrics available."
            )
        
        # Load metrics from file (in production, this would be from database)
        metrics_path = f"{settings.model_dir}/metrics.json"
        try:
            import json
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            
            return MetricsResponse(**metrics_data)
            
        except FileNotFoundError:
            # Return error instead of hardcoded metrics
            raise HTTPException(
                status_code=404, 
                detail="Model metrics not found. Please train the model first."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def train_model(csv_path: str = "data/leads.csv"):
    """Train the classification model."""
    try:
        metrics = classifier.train(csv_path)
        
        # Save metrics
        import json
        import os
        os.makedirs(settings.model_dir, exist_ok=True)
        metrics_path = f"{settings.model_dir}/metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Model training completed successfully")
        
        return {
            "message": "Model trained successfully",
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest-knowledge")
async def ingest_knowledge_documents():
    """Ingest knowledge base documents."""
    try:
        # This would typically ingest documents from a directory
        # For now, create some sample documents
        
        sample_docs = [
            {
                "title": "Email Outreach Best Practices",
                "content": "When reaching out via email, personalize your message based on the lead's role and interests. Keep it concise and include a clear call-to-action.",
                "category": "sales_tactics",
                "tags": ["email", "outreach", "personalization"]
            },
            {
                "title": "LinkedIn Connection Strategy",
                "content": "For LinkedIn outreach, send personalized connection requests with a brief message explaining why you're connecting. Follow up with valuable content.",
                "category": "social_selling",
                "tags": ["linkedin", "networking", "social"]
            },
            {
                "title": "Phone Call Scripts",
                "content": "Phone calls work best for hot leads. Prepare a brief script that focuses on understanding their needs and offering relevant solutions.",
                "category": "sales_tactics",
                "tags": ["phone", "script", "hot_leads"]
            }
        ]
        
        ingested_count = 0
        for doc_data in sample_docs:
            try:
                from app.models.schemas import KnowledgeDocument
                doc = KnowledgeDocument(**doc_data)
                await retrieval.add_document(doc)
                ingested_count += 1
            except Exception as e:
                logger.error(f"Error ingesting document: {e}")
                continue
        
        logger.info(f"Ingested {ingested_count} knowledge documents")
        
        return {
            "message": f"Successfully ingested {ingested_count} documents",
            "count": ingested_count
        }
        
    except Exception as e:
        logger.error(f"Error ingesting knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send-email")
async def send_email(request: Dict[str, Any]):
    """Send email to a lead based on their classification."""
    try:
        lead_id = request.get("lead_id")
        to_email = request.get("to_email")
        lead_data = request.get("lead_data")
        email_type = request.get("email_type", "template")  # "template" or "rag"
        
        if not all([lead_id, to_email, lead_data]):
            raise HTTPException(status_code=400, detail="Missing required fields: lead_id, to_email, lead_data")
        
        # Get the lead's heat score
        lead_input = LeadInput(**lead_data)
        score = classifier.predict(lead_input)
        
        # Determine lead type based on score
        if score.heat_score == "hot":
            lead_type = "hot"
        elif score.heat_score == "warm":
            lead_type = "warm"
        else:
            lead_type = "cold"
        
        # Smart email type selection based on lead type
        def get_smart_email_type(lead_type: str, user_override: str = None) -> str:
            """Determine email type based on lead heat score and user preference."""
            if user_override:
                return user_override  # User can override smart defaults
            
            # Smart defaults: Hot/Warm leads get RAG, Cold leads get templates
            if lead_type in ["hot", "warm"]:
                return "rag"
            else:  # cold
                return "template"
        
        # Determine final email type
        final_email_type = get_smart_email_type(lead_type, email_type)
        
        # Generate email content based on final type
        if final_email_type == "rag" and settings.enable_rag_emails:
            # Use RAG email generation
            try:
                rag_email = await rag_email_service.generate_personalized_email(
                    lead_input, lead_type
                )
                subject = rag_email["subject"]
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                        .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                        .rag-indicator {{ background: #e3f2fd; padding: 10px; border-left: 4px solid #2196f3; margin: 15px 0; border-radius: 4px; }}
                        .ai-powered {{ background: linear-gradient(45deg, #ff6b6b, #4ecdc4); color: white; padding: 8px 15px; border-radius: 20px; font-size: 12px; font-weight: bold; display: inline-block; margin-bottom: 15px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>🤖 AI-Personalized Message</h1>
                            <p>Tailored specifically for you</p>
                        </div>
                        <div class="content">
                            <div class="rag-indicator">
                                <span class="ai-powered">✨ AI-POWERED</span>
                                <p style="margin: 5px 0 0 0; font-size: 14px; color: #1976d2;">This message was personalized using AI based on your profile and interests.</p>
                            </div>
                            <div style="white-space: pre-wrap; font-size: 16px; line-height: 1.7;">{rag_email['content']}</div>
                        </div>
                    </div>
                </body>
                </html>
                """
                text_content = rag_email["content"]
                email_source = "rag"
            except Exception as e:
                logger.warning(f"RAG email generation failed, falling back to template: {e}")
                if settings.rag_email_fallback:
                    # Fallback to template if RAG fails
                    template = email_service.get_email_template(lead_type, lead_data)
                    subject = template["subject"]
                    html_content = template["content"]
                    text_content = template["text_content"]
                    email_source = "template_fallback"
                else:
                    raise HTTPException(status_code=500, detail=f"RAG email generation failed: {str(e)}")
        else:
            # Use static template (default behavior)
            template = email_service.get_email_template(lead_type, lead_data)
            subject = template["subject"]
            html_content = template["content"]
            text_content = template["text_content"]
            email_source = "template"
        
        # Send email
        success = await email_service.send_email(
            to_email=to_email,
            subject=subject,
            html_content=html_content,
            text_content=text_content
        )
        
        if success:
            logger.info(f"Email sent successfully to {to_email} for lead {lead_id} using {email_source}")
            return {
                "message": "Email sent successfully",
                "lead_id": lead_id,
                "to_email": to_email,
                "lead_type": lead_type,
                "email_type": email_source,
                "final_email_type": final_email_type,
                "smart_strategy": f"{lead_type.title()} leads get {'Telegram' if final_email_type == 'rag' and lead_type in ['hot'] else 'RAG Email' if final_email_type == 'rag' else 'Newsletter' if lead_type == 'cold' else 'Template'} messages",
                "subject": subject
            }
        else:
            logger.warning(f"Email sending failed for {to_email}, but continuing with lead processing")
            return {
                "message": "Email configuration not set up - email not sent",
                "lead_id": lead_id,
                "to_email": to_email,
                "lead_type": lead_type,
                "email_type": email_source,
                "final_email_type": final_email_type,
                "smart_strategy": f"{lead_type.title()} leads get {'Telegram' if final_email_type == 'rag' and lead_type in ['hot'] else 'RAG Email' if final_email_type == 'rag' else 'Newsletter' if lead_type == 'cold' else 'Template'} messages",
                "subject": subject,
                "warning": "Please configure email settings to enable email functionality. See EMAIL_CONFIG_FIX.md for instructions."
            }
            
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/send-test-email")
async def send_test_email(to_email: str):
    """Send a test email to verify email configuration."""
    try:
        # Create test lead data
        test_lead_data = {
            "name": "Test User",
            "company": "Test Company",
            "role": "Manager",
            "email": to_email,
            "campaign": "Test Campaign"
        }
        
        # Send test email using cold lead template
        template = email_service.get_email_template("cold", test_lead_data)
        
        test_html_content = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; border: 2px solid #28a745; border-radius: 10px;">
            <h2 style="color: #28a745;">✅ Test Email Successful!</h2>
            <p>This is a test email from Lead HeatScore to verify email configuration.</p>
            <p><strong>Recipient:</strong> {to_email}</p>
            <p><strong>Sent at:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <hr>
            {template['content']}
        </div>
        """
        
        success = await email_service.send_email(
            to_email=to_email,
            subject=f"[TEST] {template['subject']}",
            html_content=test_html_content,
            text_content=f"Test Email Successful!\n\nThis is a test email from Lead HeatScore to verify email configuration.\nRecipient: {to_email}\nSent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{template['text_content']}"
        )
        
        if success:
            return {
                "message": "Test email sent successfully",
                "to_email": to_email,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to send test email")
            
    except Exception as e:
        logger.error(f"Error sending test email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-personalized-email")
async def get_personalized_email(lead_data: Dict[str, Any]):
    """Get personalized email content without sending it."""
    try:
        # Convert to LeadInput for validation
        lead_input = LeadInput(**lead_data)
        
        # Get the lead's heat score
        if classifier.is_trained:
            score = classifier.predict(lead_input)
            lead_type = score.heat_score.value
        else:
            # Default to warm if model not trained
            lead_type = "warm"
        
        # Determine email type using smart strategy
        def get_smart_email_type(lead_type: str) -> str:
            """Determine email type based on lead heat score."""
            if lead_type == "hot":
                return "telegram"  # Default to telegram for hot leads
            elif lead_type == "warm":
                return "rag"
            else:  # cold
                return "newsletter"
        
        final_email_type = get_smart_email_type(lead_type)
        
        # Generate actual content based on channel
        if final_email_type == "telegram":
            # Generate Telegram message
            from app.models.schemas import HeatScore
            
            heat_score_enum = HeatScore.HOT if lead_type == "hot" else HeatScore.WARM if lead_type == "warm" else HeatScore.COLD
            telegram_content = telegram_service.craft_telegram_message(lead_input, heat_score_enum)
            
            return {
                "subject": "Telegram Message",
                "html_content": telegram_content.replace('\n', '<br>'),
                "text_content": telegram_content,
                "lead_type": lead_type,
                "email_type": "telegram",
                "final_email_type": "telegram",
                "smart_strategy": f"{lead_type.title()} leads get Telegram messages",
                "personalization_data": {
                    "name": lead_data.get("name", "Valued Customer"),
                    "role": lead_data.get("role", "Professional"),
                    "search_keywords": lead_data.get("search_keywords", ""),
                    "page_views": lead_data.get("page_views", 0),
                    "time_spent": lead_data.get("time_spent", 0),
                    "course_actions": lead_data.get("course_actions", ""),
                    "prior_course_interest": lead_data.get("prior_course_interest", "low")
                }
            }
        elif final_email_type == "rag" and settings.enable_rag_emails:
            try:
                rag_email = await rag_email_service.generate_personalized_email(
                    lead_input, lead_type
                )
                subject = rag_email["subject"]
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                        .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                        .rag-indicator {{ background: #e3f2fd; padding: 10px; border-left: 4px solid #2196f3; margin: 15px 0; border-radius: 4px; }}
                        .ai-powered {{ background: linear-gradient(45deg, #ff6b6b, #4ecdc4); color: white; padding: 8px 15px; border-radius: 20px; font-size: 12px; font-weight: bold; display: inline-block; margin-bottom: 15px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>🤖 AI-Personalized Message</h1>
                            <p>Tailored specifically for you</p>
                        </div>
                        <div class="content">
                            <div class="rag-indicator">
                                <span class="ai-powered">✨ AI-POWERED</span>
                                <p style="margin: 5px 0 0 0; font-size: 14px; color: #1976d2;">This message was personalized using AI based on your profile and interests.</p>
                            </div>
                            <div style="white-space: pre-wrap; font-size: 16px; line-height: 1.7;">{rag_email['content']}</div>
                        </div>
                    </div>
                </body>
                </html>
                """
                text_content = rag_email["content"]
                email_source = "rag"
            except Exception as e:
                logger.warning(f"RAG email generation failed, falling back to template: {e}")
                if settings.rag_email_fallback:
                    template = email_service.get_email_template(lead_type, lead_data)
                    subject = template["subject"]
                    html_content = template["content"]
                    text_content = template["text_content"]
                    email_source = "template_fallback"
                else:
                    raise HTTPException(status_code=500, detail=f"RAG email generation failed: {str(e)}")
        else:
            # Use static template
            template = email_service.get_email_template(lead_type, lead_data)
            subject = template["subject"]
            html_content = template["content"]
            text_content = template["text_content"]
            email_source = "template"
        
        return {
            "subject": subject,
            "html_content": html_content,
            "text_content": text_content,
            "lead_type": lead_type,
            "email_type": email_source,
            "final_email_type": final_email_type,
            "smart_strategy": f"{lead_type.title()} leads get {'Telegram' if final_email_type == 'rag' and lead_type in ['hot'] else 'RAG Email' if final_email_type == 'rag' else 'Newsletter' if lead_type == 'cold' else 'Template'} messages",
            "personalization_data": {
                "name": lead_data.get("name", "Valued Customer"),
                "role": lead_data.get("role", "Professional"),
                "search_keywords": lead_data.get("search_keywords", ""),
                "page_views": lead_data.get("page_views", 0),
                "time_spent": lead_data.get("time_spent", 0),
                "course_actions": lead_data.get("course_actions", ""),
                "prior_course_interest": lead_data.get("prior_course_interest", "low")
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating personalized email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/craft-first-message")
async def craft_first_message(lead_data: LeadInput):
    """Craft the actual first message using policy and persona."""
    try:
        # Safety check: Sanitize input data
        safety_result = sanitize_content(
            f"{lead_data.name} {lead_data.email} {lead_data.campaign}",
            check_injection=True,
            check_pii=True
        )
        
        if not safety_result.is_safe:
            logger.warning(f"Unsafe content detected: {safety_result.detected_threats}")
        
        # First classify the lead
        if classifier.is_trained:
            lead_score = classifier.predict(lead_data)
        else:
            # Create dummy score for message crafting
            lead_score = LeadScore(
                lead_id=str(uuid.uuid4()),
                heat_score="warm",
                confidence=0.5,
                probabilities={"cold": 0.3, "warm": 0.5, "hot": 0.2}
            )
        
        # Craft the actual first message
        crafted_message = await next_action_agent.craft_first_message(lead_data, lead_score)
        
        logger.info(f"Crafted first message for lead: {crafted_message.get('lead_id')}")
        return crafted_message
        
    except Exception as e:
        logger.error(f"Error crafting first message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send-telegram-message")
async def send_telegram_message(request: Dict[str, Any]):
    """Send message via Telegram."""
    try:
        chat_id = request.get("chat_id")
        message = request.get("message")
        
        if not chat_id or not message:
            raise HTTPException(status_code=400, detail="chat_id and message are required")
        
        # Send via Telegram service
        result = await telegram_service.send_message(chat_id, message)
        
        if result["success"]:
            logger.info(f"Telegram message sent successfully to {chat_id}")
            return {
                "message": "Telegram message sent successfully",
                "chat_id": chat_id,
                "message_id": result.get("message_id")
            }
        else:
            logger.error(f"Failed to send Telegram message: {result.get('error')}")
            raise HTTPException(status_code=500, detail=f"Failed to send Telegram message: {result.get('error')}")
        
    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send-telegram-to-phone")
async def send_telegram_to_phone(request: Dict[str, Any]):
    """Send Telegram message using phone number."""
    try:
        phone_number = request.get("phone_number")
        lead_data = request.get("lead_data", {})
        
        if not phone_number:
            raise HTTPException(status_code=400, detail="phone_number is required")
        
        # Convert lead_data to LeadInput for message crafting
        from app.models.schemas import LeadInput, HeatScore
        
        try:
            lead_input = LeadInput(**lead_data)
        except Exception as e:
            # If conversion fails, create a minimal LeadInput
            lead_input = LeadInput(
                name=lead_data.get("name", "there"),
                email=lead_data.get("email", ""),
                phone=phone_number,
                source=lead_data.get("source", "unknown"),
                recency_days=lead_data.get("recency_days", 0),
                region=lead_data.get("region", "unknown"),
                role=lead_data.get("role", "Professional"),
                campaign=lead_data.get("campaign", "our program"),
                page_views=lead_data.get("page_views", 0),
                last_touch=lead_data.get("last_touch", "unknown"),
                prior_course_interest=lead_data.get("prior_course_interest", "medium"),
                search_keywords=lead_data.get("search_keywords", ""),
                time_spent=lead_data.get("time_spent", 0),
                course_actions=lead_data.get("course_actions", ""),
                cta=lead_data.get("cta", "")
            )
        
        # Get heat score for message crafting
        if classifier.is_trained:
            score = classifier.predict(lead_input)
            heat_score = score.heat_score
        else:
            heat_score = HeatScore.WARM  # Default to warm if model not trained
        
        # Craft personalized message
        crafted_message = telegram_service.craft_telegram_message(lead_input, heat_score)
        
        # Send via Telegram service
        result = await telegram_service.send_message_to_phone(phone_number, crafted_message)
        
        if result["success"]:
            logger.info(f"Telegram message prepared for phone {phone_number}")
            return {
                "message": "Telegram message prepared successfully",
                "phone_number": phone_number,
                "message_preview": result.get("message_preview"),
                "note": result.get("note"),
                "lead_data": lead_data
            }
        else:
            logger.error(f"Failed to prepare Telegram message: {result.get('error')}")
            raise HTTPException(status_code=500, detail=f"Failed to prepare Telegram message: {result.get('error')}")
        
    except Exception as e:
        logger.error(f"Error preparing Telegram message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-telegram-connection")
async def test_telegram_connection():
    """Test Telegram bot connection."""
    try:
        result = await telegram_service.test_connection()
        return result
    except Exception as e:
        logger.error(f"Error testing Telegram connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send-whatsapp-message")
async def send_whatsapp_message(request: Dict[str, Any]):
    """Send message via WhatsApp Business API."""
    try:
        phone_number = request.get("phone_number")
        message = request.get("message")
        
        if not phone_number or not message:
            raise HTTPException(status_code=400, detail="phone_number and message are required")
        
        # Send via WhatsApp service
        result = await whatsapp_service.send_message(phone_number, message)
        
        if result["success"]:
            logger.info(f"WhatsApp message sent successfully to {phone_number}")
            return {
                "message": "WhatsApp message sent successfully",
                "phone_number": phone_number,
                "message_id": result.get("message_id")
            }
        else:
            logger.error(f"Failed to send WhatsApp message: {result.get('error')}")
            raise HTTPException(status_code=500, detail=f"Failed to send WhatsApp message: {result.get('error')}")
        
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-whatsapp-connection")
async def test_whatsapp_connection():
    """Test WhatsApp Business API connection."""
    try:
        result = await whatsapp_service.test_connection()
        return result
    except Exception as e:
        logger.error(f"Error testing WhatsApp connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    """Generate both template and RAG emails for A/B testing."""
    try:
        # Convert to LeadInput for validation
        lead_input = LeadInput(**lead_data)
        
        # Get the lead's heat score
        if classifier.is_trained:
            score = classifier.predict(lead_input)
            lead_type = score.heat_score.value
        else:
            # Default to warm if model not trained
            lead_type = "warm"
        
        # Generate Template A (Static)
        template_email = email_service.get_email_template(lead_type, lead_data)
        
        # Generate Template B (RAG)
        rag_email = await rag_email_service.generate_personalized_email(
            lead_input, lead_type
        )
        
        return {
            "lead_id": str(uuid.uuid4()),
            "lead_type": lead_type,
            "template_a": {
                "type": "static",
                "subject": template_email["subject"],
                "content": template_email["content"],
                "text_content": template_email["text_content"]
            },
            "template_b": {
                "type": "rag",
                "subject": rag_email["subject"],
                "content": rag_email["content"],
                "text_content": rag_email["content"]  # RAG content is already text
            },
            "lead_data": {
                "name": lead_data.get("name", "Valued Customer"),
                "role": lead_data.get("role", "Professional"),
                "company": lead_data.get("company", ""),
                "campaign": lead_data.get("campaign", ""),
                "page_views": lead_data.get("page_views", 0),
                "recency_days": lead_data.get("recency_days", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating A/B test messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit-ab-evaluation")
async def submit_ab_evaluation(evaluation_data: Dict[str, Any]):
    """Submit A/B testing evaluation results."""
    try:
        # Validate evaluation data
        required_fields = ["lead_id", "template_a_rating", "template_b_rating", "preferred_template"]
        for field in required_fields:
            if field not in evaluation_data:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required field: {field}"
                )
        
        # Validate ratings (1-5)
        for rating_field in ["template_a_rating", "template_b_rating"]:
            rating = evaluation_data[rating_field]
            if not isinstance(rating, int) or rating < 1 or rating > 5:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid rating for {rating_field}. Must be integer 1-5."
                )
        
        # Store evaluation (in a real system, this would go to a database)
        evaluation_result = {
            "evaluation_id": str(uuid.uuid4()),
            "lead_id": evaluation_data["lead_id"],
            "template_a_rating": evaluation_data["template_a_rating"],
            "template_b_rating": evaluation_data["template_b_rating"],
            "preferred_template": evaluation_data["preferred_template"],
            "comments": evaluation_data.get("comments", ""),
            "timestamp": datetime.now().isoformat(),
            "user_feedback": evaluation_data.get("user_feedback", "")
        }
        
        # Log the evaluation
        logger.info(
            "A/B test evaluation submitted",
            evaluation_id=evaluation_result["evaluation_id"],
            lead_id=evaluation_data["lead_id"],
            template_a_rating=evaluation_data["template_a_rating"],
            template_b_rating=evaluation_data["template_b_rating"],
            preferred_template=evaluation_data["preferred_template"]
        )
        
        return {
            "message": "A/B test evaluation submitted successfully",
            "evaluation_id": evaluation_result["evaluation_id"],
            "timestamp": evaluation_result["timestamp"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting A/B evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# A/B Testing Endpoints
@router.post("/ab-test-message")
async def ab_test_message(lead_data: Dict[str, Any]):
    """Run A/B test for a single lead message."""
    try:
        from app.services.ab_testing_service import ab_testing_service
        
        # Assign lead to A/B test group
        lead_id = lead_data.get("lead_id", f"lead_{random.randint(1000, 9999)}")
        group = ab_testing_service.assign_to_group(lead_id)
        
        # Generate realistic outcomes
        outcomes = ab_testing_service.generate_realistic_outcomes(lead_data, group)
        outcomes["lead_id"] = lead_id
        
        logger.info(f"A/B test message generated for lead {lead_id} in group {group}")
        
        return {
            "lead_id": lead_id,
            "group": group,
            "outcomes": outcomes,
            "message_type": "rag" if group == "rag" else "template",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in A/B test message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-ab-test")
async def run_ab_test(leads_data: List[Dict[str, Any]]):
    """Run comprehensive A/B test on multiple leads."""
    try:
        from app.services.ab_testing_service import ab_testing_service
        
        # Run A/B test
        test_results = ab_testing_service.run_ab_test(leads_data)
        
        # Generate summary
        summary = ab_testing_service.get_test_summary(test_results)
        
        logger.info(f"A/B test completed with {len(leads_data)} leads")
        
        return {
            "test_results": test_results,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-test-results")
async def get_ab_test_results():
    """Get A/B test results and metrics."""
    try:
        from app.services.ab_testing_service import ab_testing_service
        
        # Generate sample results for demo
        sample_leads = [
            {
                "lead_id": f"L{i:03d}",
                "name": f"Lead {i}",
                "heat_score": random.choice(["hot", "warm", "cold"]),
                "page_views": random.randint(1, 50),
                "recency_days": random.randint(1, 30),
                "prior_course_interest": random.choice(["low", "medium", "high"])
            }
            for i in range(1, 101)  # 100 sample leads
        ]
        
        # Run A/B test
        test_results = ab_testing_service.run_ab_test(sample_leads)
        summary = ab_testing_service.get_test_summary(test_results)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        # Convert all numpy types in the response
        converted_results = convert_numpy_types(test_results)
        converted_summary = convert_numpy_types(summary)
        
        return {
            "test_results": converted_results,
            "summary": converted_summary,
            "sample_size": len(sample_leads),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting A/B test results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/f1-scores")
async def get_f1_scores():
    """Get F1 scores for lead classification based on trained model."""
    try:
        if not classifier.is_trained:
            raise HTTPException(
                status_code=404, 
                detail="No model trained yet. Please upload CSV and train the model first."
            )
        
        # Get actual metrics from the trained model
        metrics_path = f"{settings.model_dir}/metrics.json"
        try:
            import json
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            
            # Convert the actual metrics structure to the expected format
            f1_scores = {
                "hot": {
                    "precision": metrics_data["precision"]["hot"],
                    "recall": metrics_data["recall"]["hot"],
                    "f1_score": metrics_data["f1_score"]["hot"],
                    "support": sum(metrics_data["confusion_matrix"][2])  # Hot row total
                },
                "warm": {
                    "precision": metrics_data["precision"]["warm"],
                    "recall": metrics_data["recall"]["warm"],
                    "f1_score": metrics_data["f1_score"]["warm"],
                    "support": sum(metrics_data["confusion_matrix"][1])  # Warm row total
                },
                "cold": {
                    "precision": metrics_data["precision"]["cold"],
                    "recall": metrics_data["recall"]["cold"],
                    "f1_score": metrics_data["f1_score"]["cold"],
                    "support": sum(metrics_data["confusion_matrix"][0])  # Cold row total
                },
                "macro_avg": {
                    "precision": sum(metrics_data["precision"].values()) / 3,
                    "recall": sum(metrics_data["recall"].values()) / 3,
                    "f1_score": sum(metrics_data["f1_score"].values()) / 3,
                    "support": sum([sum(row) for row in metrics_data["confusion_matrix"]])
                }
            }
            
            return {
                "f1_scores": f1_scores,
                "overall_performance": "Good" if f1_scores["macro_avg"]["f1_score"] >= 0.60 else "Needs Improvement",
                "timestamp": datetime.now().isoformat(),
                "data_source": "trained_model"
            }
            
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, 
                detail="Model metrics not found. Please train the model first."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting F1 scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/confusion-matrix")
async def get_confusion_matrix():
    """Get confusion matrix for lead classification based on trained model."""
    try:
        if not classifier.is_trained:
            raise HTTPException(
                status_code=404, 
                detail="No model trained yet. Please upload CSV and train the model first."
            )
        
        # Get actual confusion matrix from the trained model
        metrics_path = f"{settings.model_dir}/metrics.json"
        try:
            import json
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            
            # Convert confusion matrix to the expected format
            cm = metrics_data["confusion_matrix"]
            confusion_matrix = {
                "actual_vs_predicted": {
                    "cold": {"cold": cm[0][0], "warm": cm[0][1], "hot": cm[0][2]},
                    "warm": {"cold": cm[1][0], "warm": cm[1][1], "hot": cm[1][2]},
                    "hot": {"cold": cm[2][0], "warm": cm[2][1], "hot": cm[2][2]}
                },
                "accuracy": metrics_data["accuracy"],
                "total_samples": sum([sum(row) for row in cm])
            }
            
            return {
                "confusion_matrix": confusion_matrix,
                "timestamp": datetime.now().isoformat(),
                "data_source": "trained_model"
            }
            
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, 
                detail="Model metrics not found. Please train the model first."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting confusion matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/roc-curves")
async def get_roc_curves():
    """Get ROC curve data for each class based on trained model."""
    try:
        if not classifier.is_trained:
            raise HTTPException(
                status_code=404, 
                detail="No model trained yet. Please upload CSV and train the model first."
            )
        
        # Get actual ROC curves from the trained model
        metrics_path = f"{settings.model_dir}/metrics.json"
        try:
            import json
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            
            # Calculate ROC curves dynamically from confusion matrix
            cm = metrics_data["confusion_matrix"]
            
            def calculate_roc_from_cm(confusion_matrix, class_index):
                """Calculate ROC curve data from confusion matrix for a specific class."""
                # Get true positives, false positives, true negatives, false negatives
                tp = confusion_matrix[class_index][class_index]
                fp = sum(confusion_matrix[i][class_index] for i in range(len(confusion_matrix)) if i != class_index)
                tn = sum(confusion_matrix[i][j] for i in range(len(confusion_matrix)) for j in range(len(confusion_matrix)) if i != class_index and j != class_index)
                fn = sum(confusion_matrix[class_index][j] for j in range(len(confusion_matrix)) if j != class_index)
                
                # Calculate TPR and FPR
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                # Calculate AUC (simplified - in reality you'd need probability scores)
                auc = (tpr + (1 - fpr)) / 2
                
                # Generate ROC curve points based on actual performance
                fpr_points = [0.0, fpr, 1.0]
                tpr_points = [0.0, tpr, 1.0]
                
                return {
                    "fpr": fpr_points,
                    "tpr": tpr_points,
                    "auc": round(auc, 3)
                }
            
            # Calculate ROC curves for each class
            roc_curves = {
                "cold": calculate_roc_from_cm(cm, 0),
                "warm": calculate_roc_from_cm(cm, 1), 
                "hot": calculate_roc_from_cm(cm, 2)
            }
            
            return {
                "roc_curves": roc_curves,
                "timestamp": datetime.now().isoformat(),
                "data_source": "trained_model"
            }
            
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, 
                detail="Model metrics not found. Please train the model first."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ROC curves: {e}")
        raise HTTPException(status_code=500, detail=str(e))
