from motor.motor_asyncio import AsyncIOMotorClient
from app.utilities.logger import logger
from app.core.config import settings

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

mongodb = MongoDB()

async def connect_to_mongo():
    try:
        mongodb.client = AsyncIOMotorClient(settings.mongo_uri)
        mongodb.db = mongodb.client[settings.mongo_db_name]
        logger.info("Connected to MongoDB successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

async def close_mongo_connection():
    try:
        mongodb.client.close()
        logger.info("MongoDB connection closed.")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}")
