import os
from supabase import create_client, Client
import openai
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from dotenv import load_dotenv
from datetime import datetime
import hashlib
import aiofiles
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

load_dotenv(dotenv_path=".env.local")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for Supabase and OpenAI
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def create_memory_filename(content: str, user: str = "default") -> str:
    """
    Creates a unique filename for a memory based on content and user.
    
    Args:
        content (str): The memory content
        user (str): The user identifier
        
    Returns:
        str: A unique filename in the format: mem_[timestamp]_[user_hash]_[content_hash].txt
    """
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create short hashes for user and content
    user_hash = hashlib.md5(user.encode()).hexdigest()[:6]
    content_hash = hashlib.md5(content.encode()).hexdigest()[:6]
    
    # Create filename with max length of 100 characters
    filename = f"mem_{timestamp}_{user_hash}_{content_hash}.txt"
    return filename

async def embed_and_store(
    content: str, 
    user: str = "default", 
    chunk_size: int = 1000, 
    chunk_overlap: int = 100
) -> Dict[str, Any]:
    """
    Embeds and stores content into the Supabase 'memories' table with chunking.
    Creates a file in the memories directory to store the content.

    Args:
        content (str): The content to embed and store.
        user (str): The user identifier. Defaults to "default".
        chunk_size (int): Size of text chunks for splitting. Default is 1000 characters.
        chunk_overlap (int): Overlap between chunks. Default is 100 characters.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - filename (str): The name of the created file
            - chunks_processed (int): Number of chunks successfully processed
            - error (Optional[str]): Error message if any
    """
    result = {
        "success": False,
        "filename": "",
        "chunks_processed": 0,
        "error": None
    }
    
    try:
        # Create a unique filename
        filename = await create_memory_filename(content, user)
        result["filename"] = filename
        
        # Create the directory and file path using os.path.join
        memories_dir = "memories"
        os.makedirs(memories_dir, exist_ok=True)
        memory_file = os.path.join(memories_dir, filename)
        logger.info(f"Creating memory file at: {memory_file}")
        
        # # Ensure the memories directory exists
        # memories_dir.mkdir(parents=True, exist_ok=True)
        # logger.info(f"Using memories directory: {memories_dir}")
        
        # # Create the file path
        # memory_file = memories_dir / filename
        # logger.info(f"Creating memory file at: {memory_file}")
        
        # Write content to the file asynchronously
        try:
            async with aiofiles.open(memory_file, "w", encoding="utf-8") as f:
                await f.write(content)
            logger.info(f"Successfully created memory file: {filename}")
        except Exception as e:
            logger.error(f"Failed to write to file {memory_file}: {e}")
            raise

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Split text into chunks
        chunks = text_splitter.split_text(content)
        logger.info(f"Split content into {len(chunks)} chunks")

        # Process chunks sequentially to avoid the unhashable type error
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding for each chunk
                embedding = await asyncio.to_thread(embeddings.embed_query, chunk)
                logger.info(f"Generated embedding for chunk {i + 1}")

                # Insert chunk into the Supabase database
                response = supabase.table("memories").insert({
                    "content": chunk,
                    "embedding": embedding,
                    "source_file": filename,
                    "user": user
                }).execute()
                
                if response.data:
                    result["chunks_processed"] += 1
                    logger.info(f"Successfully stored chunk {i + 1}")
                else:
                    logger.error(f"Failed to store chunk {i + 1}: No data in response")

            except Exception as e:
                logger.error(f"Error embedding or storing chunk {i + 1}: {e}")
                result["error"] = str(e)

        result["success"] = True
        logger.info(f"Successfully processed all chunks. Total chunks: {result['chunks_processed']}")

    except Exception as e:
        logger.error(f"Error processing content: {e}")
        result["error"] = str(e)
    
    return result

async def retrieve_memories(query: str) -> Dict[str, Any]:
    """
    Retrieve relevant memories using semantic search.

    Args:
        query (str): The search query to find relevant memories

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - memories (List[Dict]): List of retrieved memories with their metadata
            - error (Optional[str]): Error message if any
    """
    result = {
        "success": False,
        "memories": [],
        "error": None
    }

    try:
        # Generate embedding for the query
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        query_vector = await asyncio.to_thread(embeddings.embed_query, query)
        
        # Perform semantic search in the 'memories' table using hybrid_search
        response = supabase.rpc(
            "hybrid_search",
            {
                "query_text": query,
                "query_embedding": query_vector,
                "match_count": 10,
                "semantic_weight": 0.7,  # Weight for semantic search
                "full_text_weight": 0.3,  # Weight for text search
                "rrf_k": 60  # Reciprocal Rank Fusion parameter
            }
        ).execute()

        if response.data:
            # Process and format the retrieved memories
            memories = []
            for item in response.data:
                memory = {
                    "content": item["content"],
                    "source_file": item["source_file"],
                    "user": item["user"],
                    "similarity_score": item.get("similarity", 0),
                    "created_at": item.get("created_at")
                }
                memories.append(memory)
            
            result["memories"] = memories
            result["success"] = True
            logger.info(f"Successfully retrieved {len(memories)} memories")
        else:
            logger.info("No relevant memories found")
            result["success"] = True  # Still successful, just no results

    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        result["error"] = str(e)

    return result