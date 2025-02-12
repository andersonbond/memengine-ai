import os
import asyncio
from datetime import datetime
import logging

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from supabase import create_client, Client

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=".env.local")

async def retrieve_policies(query: str) -> str:
    """Retrieve policy-related data from Supabase using embeddings."""
    logger.info(f"Starting policy retrieval: {query}")
    start_time = datetime.now()
    try:
        # Simulate a delay (if needed)
        await asyncio.sleep(3)
        
        # Generate embedding for the query
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        query_vector = embeddings.embed_query(query)
        
        # Perform a similarity search in the 'memories' table
        response = supabase.rpc(
            "hybrid_search",
            {
                "query_text": query,
                "query_embedding": query_vector,
                "match_count": 7,
            }
        ).execute()
        await asyncio.sleep(2)
        if response.data:
            # Concatenate the retrieved policy data
            policies = "\n".join([item['content'] for item in response.data])
            logger.info(f"Policy retrieval completed in {datetime.now() - start_time}")
            return policies
        else:
            logger.info(f"Policy retrieval completed in {datetime.now() - start_time}")
            return "No relevant policies found."
    except Exception as e:
        logger.error(f"Error retrieving policies: {str(e)}")
        return "An error occurred while retrieving policies."