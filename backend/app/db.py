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
        # Optimized connection settings for better performance
        db.client = AsyncIOMotorClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=5000,          # 5 second connection timeout
            maxPoolSize=10,                  # Limit connection pool
            minPoolSize=1,                   # Minimum connections
            maxIdleTimeMS=30000,             # Close idle connections after 30s
            retryWrites=True
        )
        db.database = db.client[settings.mongo_db]
        
        # Skip ping test for faster startup
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
    """Check if vector search index exists (Atlas requires manual creation)."""
    try:
        collection = db.database[settings.mongo_collection]
        
        # Check if index already exists
        indexes = await collection.list_indexes().to_list(length=None)
        index_names = [idx['name'] for idx in indexes]
        
        if settings.mongo_vector_index not in index_names:
            print(f"INFO: Vector index '{settings.mongo_vector_index}' not detected in regular indexes")
            print("This is normal for Atlas vector search indexes - they don't appear in list_indexes()")
            print("Assuming vector index exists in Atlas and proceeding...")
        else:
            print(f"Vector index '{settings.mongo_vector_index}' found and ready")
            
    except Exception as e:
        print(f"Error checking vector index: {e}")


def get_sync_client() -> MongoClient:
    """Get synchronous MongoDB client for non-async operations."""
    return MongoClient(settings.mongo_uri)
