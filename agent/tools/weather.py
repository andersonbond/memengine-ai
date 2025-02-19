from typing import Annotated
import aiohttp
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

async def get_weather(location: str) -> str:
        """Called when the user asks about the weather. This function will return the weather for the given location."""
        logger.info(f"getting weather for {location}")
        url = f"https://wttr.in/{location}?format=%C+%t"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    weather_data = await response.text()
                    # response from the function call is returned to the LLM
                    return f"The weather in {location} is {weather_data}."
                else:
                    raise Exception(
                        f"Failed to get weather data, status code: {response.status}"
                    )