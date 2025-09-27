#!/usr/bin/env python3
"""
Script to add knowledge documents to MongoDB for RAG testing
"""

import asyncio
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

async def add_knowledge_documents():
    """Add sample knowledge documents to MongoDB."""
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.mongo_uri)
    db = client[settings.mongo_db]
    collection = db[settings.mongo_collection]
    
    # Sample knowledge documents
    knowledge_docs = [
        {
            "title": "Data Science Program Overview",
            "content": "Our Data Science program covers Python programming, machine learning algorithms, statistical analysis, data visualization with Power BI, and real-world projects. Perfect for managers and working professionals looking to advance their careers in analytics and data-driven decision making.",
            "category": "course_info",
            "tags": ["data-science", "python", "machine-learning", "power-bi", "analytics"],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "embedding": [0.1] * 1536  # Dummy embedding
        },
        {
            "title": "AI Course Benefits",
            "content": "The AI Course provides hands-on experience with neural networks, deep learning, and AI applications. Students and professionals can learn cutting-edge AI technologies including natural language processing, computer vision, and predictive analytics.",
            "category": "course_info",
            "tags": ["ai", "neural-networks", "deep-learning", "nlp", "computer-vision"],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "embedding": [0.2] * 1536
        },
        {
            "title": "Business Analytics Program",
            "content": "Business Analytics focuses on SQL, reporting, business intelligence, and data-driven decision making. Ideal for managers who need to make strategic decisions and working professionals in business roles looking to enhance their analytical skills.",
            "category": "course_info",
            "tags": ["business-analytics", "sql", "reporting", "business-intelligence", "decision-making"],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "embedding": [0.3] * 1536
        },
        {
            "title": "Manager Career Development",
            "content": "Managers can leverage our programs to enhance their analytical skills, make data-driven decisions, and lead data science teams. Our courses provide practical tools for business intelligence, reporting, strategic planning, and team management.",
            "category": "career_guidance",
            "tags": ["managers", "leadership", "business-intelligence", "strategic-planning", "team-management"],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "embedding": [0.4] * 1536
        },
        {
            "title": "Student Learning Path",
            "content": "Students can start with fundamentals and progress to advanced topics. Our programs include hands-on projects, mentorship, industry-relevant case studies, and career preparation to help students transition into tech and analytics careers.",
            "category": "student_resources",
            "tags": ["students", "learning-path", "projects", "mentorship", "career-prep"],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "embedding": [0.5] * 1536
        },
        {
            "title": "Working Professional Benefits",
            "content": "Working professionals can upskill and advance their careers with our industry-relevant programs. We offer flexible learning schedules, practical projects, and career support to help professionals transition into data science and analytics roles.",
            "category": "career_guidance",
            "tags": ["working-professionals", "upskilling", "career-advancement", "flexible-learning"],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "embedding": [0.6] * 1536
        },
        {
            "title": "Career Support Services",
            "content": "We provide comprehensive career guidance, placement support, mentorship programs, industry-recognized certificates, and job placement assistance. Our team helps with resume building, interview preparation, and connecting students with industry opportunities.",
            "category": "support",
            "tags": ["career-guidance", "placement", "mentorship", "certification", "job-support"],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "embedding": [0.7] * 1536
        },
        {
            "title": "Learning Resources and Projects",
            "content": "Access to practical hands-on projects, real-world case studies, expert mentorship, industry partnerships, and cutting-edge tools. Our resources are designed for different learning levels and career goals, from beginners to advanced professionals.",
            "category": "resources",
            "tags": ["hands-on", "projects", "case-studies", "mentorship", "industry-partnerships"],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "embedding": [0.8] * 1536
        }
    ]
    
    try:
        # Clear existing documents first
        await collection.delete_many({})
        print("Cleared existing documents")
        
        # Insert new documents
        result = await collection.insert_many(knowledge_docs)
        print(f"Successfully added {len(result.inserted_ids)} knowledge documents")
        
        # Verify the count
        count = await collection.count_documents({})
        print(f"Total documents in collection: {count}")
        
    except Exception as e:
        print(f"Error adding documents: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(add_knowledge_documents())
