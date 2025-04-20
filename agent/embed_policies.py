import os
import glob
from supabase import create_client, Client
import openai
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from dotenv import load_dotenv

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

def read_files_from_folder(folder_path: str):
    """Reads all text files from the specified folder."""
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    file_contents = {}
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            file_contents[file] = f.read()
    return file_contents

def embed_and_store_policies(folder_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Embeds and stores text files into the Supabase 'policies' table with chunking.

    Args:
        folder_path (str): Path to the folder containing text files.
        chunk_size (int): Size of text chunks for splitting. Default is 500 characters.
        chunk_overlap (int): Overlap between chunks. Default is 50 characters.
    """
    try:
        # Load all files from the folder
        files = read_files_from_folder(folder_path)
        logger.info(f"Found {len(files)} files in folder: {folder_path}")

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for file_name, content in files.items():
            logger.info(f"Processing file: {file_name}")
            
            # Split text into chunks
            chunks = text_splitter.split_text(content)

            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding for each chunk
                    embedding = embeddings.embed_query(chunk)

                    # Insert chunk into the Supabase database
                    supabase.table("policies").insert({
                        "content": chunk,
                        "embedding": embedding,
                    }).execute()

                    logger.info(f"Successfully stored chunk {i + 1} of {file_name}")

                except Exception as e:
                    logger.error(f"Error embedding or storing chunk {i + 1} of {file_name}: {e}")

    except Exception as e:
        logger.error(f"Error processing folder: {e}")

if __name__ == "__main__":
    embed_and_store_policies(folder_path="./policies")