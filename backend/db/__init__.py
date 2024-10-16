import os
from pymongo import MongoClient
from pymongo.database import Database, Collection
from pymongo.errors import DuplicateKeyError
from loguru import logger

def connect_to_mongo():
    """
    This function establishes a connection to the MongoDB database using the URL
    provided in the environment variable 'MONGO_DB_URL'.
    Returns:
        Database: The MongoDB database object.
    Raises:
        Exception: If there is an error connecting to the MongoDB database.
    """
    client = MongoClient(os.getenv("MONGO_DB_URL"))
    database = client["development"]

    try:
        client.admin.command("ping")
        logger.success("### Database is Connected Successfully! ###")
    except Exception as e:
        raise Exception("The following error occurred: ", e)
    
    # Create indexes for users collection
    users_collection = get_collection(database, "users")
    try:
        users_collection.create_index("username", unique=True)
        users_collection.create_index("email", unique=True)
        logger.info("### Indexes created for 'users' collection ###")
    except DuplicateKeyError as e:
        logger.warning(f"### Index already exists for 'users' collection: {e} ###")
    
    # Create indexes for sentiment_history collection
    sentiment_collection = get_collection(database, "sentiment_history")
    try:
        sentiment_collection.create_index("username")
        sentiment_collection.create_index("timestamp")
        logger.info("### Indexes created for 'sentiment_history' collection ###")
    except DuplicateKeyError as e:
        logger.warning(f"### Index already exists for 'sentiment_history' collection: {e} ###")
    
    return database


def get_collection(database: Database, collection_name: str) -> Collection:
    """
    This function retrieves a collection from the specified database.
    Args:
        database (Database): The database object.
        collection_name (str): The name of the collection to retrieve.
    Returns:
        Collection: The collection object. 
    """
    collection = database[f"{collection_name}"]
    return collection
