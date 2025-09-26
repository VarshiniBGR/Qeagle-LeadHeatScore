from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from app.config import settings
import asyncio
from typing import Optional


class Database:
    client: Optional[AsyncIOMotorClient] = None
    database: Optional[AsyncIOMotorDatabase] = None


db = Database()


async def get_database() -> AsyncIOMotorDatabase:
    """Get database instance."""
    return db.database


async def connect_to_mongo():
    """Create database connection."""
    try:
        db.client = AsyncIOMotorClient(settings.mongo_uri)
        db.database = db.client[settings.mongo_db]
        
        # Test connection
        await db.client.admin.command('ping')
        print(f"Connected to MongoDB: {settings.mongo_db}")
        
        # Create vector search index if it doesn't exist
        await create_vector_index()
        
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise


async def close_mongo_connection():
    """Close database connection."""
    if db.client:
        db.client.close()


async def create_vector_index():
    """Create vector search index for embeddings."""
    try:
        collection = db.database[settings.mongo_collection]
        
        # Check if index already exists
        indexes = await collection.list_indexes().to_list(length=None)
        index_names = [idx['name'] for idx in indexes]
        
        if settings.mongo_vector_index not in index_names:
            # Create vector search index
            index_definition = {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 384,  # all-MiniLM-L6-v2 dimensions
                        "similarity": "cosine"
                    }
                ]
            }
            
            await collection.create_index(
                index_definition,
                name=settings.mongo_vector_index
            )
            print(f"Created vector index: {settings.mongo_vector_index}")
        else:
            print(f"Vector index already exists: {settings.mongo_vector_index}")
            
    except Exception as e:
        print(f"Error creating vector index: {e}")


def get_sync_client() -> MongoClient:
    """Get synchronous MongoDB client for non-async operations."""
    return MongoClient(settings.mongo_uri)
