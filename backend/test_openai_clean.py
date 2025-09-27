#!/usr/bin/env python3
"""
Clean OpenAI Integration Test
Tests the simplified OpenAI-only setup for Lead HeatScore system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.config import settings
from app.services.rag_email_service import rag_email_service
from app.services.retrieval import retrieval
from app.models.schemas import LeadInput

async def test_openai_integration():
    """Test OpenAI integration components."""
    print("🚀 Testing OpenAI Integration")
    print("=" * 50)
    
    # Test 1: Configuration Check
    print("\n1️⃣ Configuration Check")
    print(f"OpenAI API Key: {'✅ Set' if settings.openai_api_key else '❌ Missing'}")
    print(f"LLM Model: {settings.llm_model}")
    print(f"Embedding Model: {settings.embedding_model_name}")
    print(f"Max Tokens: {settings.llm_max_tokens}")
    print(f"Temperature: {settings.llm_temperature}")
    
    # Test 2: RAG Email Service Initialization
    print("\n2️⃣ RAG Email Service Test")
    try:
        print(f"LLM Type: {rag_email_service.llm_type}")
        print(f"LLM Available: {'✅ Yes' if rag_email_service.llm else '❌ No'}")
        
        if rag_email_service.llm:
            print("✅ OpenAI LLM initialized successfully")
        else:
            print("❌ OpenAI LLM not initialized")
            
    except Exception as e:
        print(f"❌ Error testing RAG service: {e}")
    
    # Test 3: Embedding Generation Test
    print("\n3️⃣ Embedding Generation Test")
    try:
        test_text = "This is a test for OpenAI embedding generation"
        embedding = retrieval._compute_embedding(test_text)
        
        print(f"Embedding Dimensions: {len(embedding)}")
        print(f"Embedding Sample: {embedding[:5]}...")
        
        if len(embedding) > 0 and embedding[0] != 0.0:
            print("✅ OpenAI embeddings working correctly")
        else:
            print("❌ OpenAI embeddings not working (using dummy)")
            
    except Exception as e:
        print(f"❌ Error testing embeddings: {e}")
    
    # Test 4: Email Generation Test
    print("\n4️⃣ Email Generation Test")
    try:
        # Create test lead data
        test_lead = LeadInput(
            name="John Doe",
            email="john@example.com",
            source="Web",
            recency_days=3,
            region="North America",
            role="Software Engineer",
            company="Tech Corp",
            campaign="AI Course",
            page_views=15,
            last_touch="Email Open",
            prior_course_interest="high"
        )
        
        # Generate email
        email_result = await rag_email_service.generate_personalized_email(
            lead_data=test_lead,
            lead_type="hot",
            force_template=False  # Use OpenAI
        )
        
        print(f"Email Type: {email_result.get('type', 'unknown')}")
        print(f"Subject: {email_result.get('subject', 'N/A')[:50]}...")
        print(f"Content Length: {len(email_result.get('content', ''))} characters")
        
        if email_result.get('type') == 'rag':
            print("✅ OpenAI email generation working")
        else:
            print("⚠️ Using fallback email generation")
            
    except Exception as e:
        print(f"❌ Error testing email generation: {e}")
    
    # Test 5: Model Switching Test
    print("\n5️⃣ Model Switching Test")
    try:
        # Test demo mode (GPT-4o)
        rag_email_service.switch_to_demo_mode()
        print(f"Demo Mode: {rag_email_service.llm_type}")
        
        # Test production mode (GPT-4o-mini)
        rag_email_service.switch_to_production_mode()
        print(f"Production Mode: {rag_email_service.llm_type}")
        
        print("✅ Model switching working correctly")
        
    except Exception as e:
        print(f"❌ Error testing model switching: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 OpenAI Integration Test Complete")
    print("\nNext Steps:")
    print("1. Update your .env file with your OpenAI API key")
    print("2. Restart the backend server")
    print("3. Test the API endpoints")
    print("4. Monitor the logs for OpenAI usage")

if __name__ == "__main__":
    asyncio.run(test_openai_integration())
