import logging
import os
from typing import List, Annotated
from datetime import datetime
import random
import re
import urllib
import aiohttp
import asyncio

from dotenv import load_dotenv

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    multimodal
)
from livekit.agents.pipeline import AgentCallContext, VoicePipelineAgent
from livekit.agents.multimodal import MultimodalAgent

from livekit.plugins import openai, deepgram, silero
from langchain_openai.embeddings import OpenAIEmbeddings
from supabase import create_client, Client

# Load environment variables
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def prewarm(proc: JobProcess):
    """Pre-warm resources like VAD for faster startup."""
    proc.userdata["vad"] = silero.VAD.load()

async def retrieve_policies(query: str) -> str:
    """Retrieve policy-related data from Supabase using embeddings."""
    logger.info(f"retrieve_policies_function called with query: {query}")

    try:
        # Generate embedding for the query
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        query_vector = embeddings.embed_query(query)
        
        # Perform a similarity search in the 'memories' table
        response = supabase.rpc(
            "hybrid_search",
            {
                "query_text": query,
                "query_embedding": query_vector,
                "match_count": 5,
            }
        ).execute()

        if response.data:
            # Concatenate the retrieved policy data
            policies = "\n".join([item['content'] for item in response.data])
            return policies
        else:
            return "No relevant policies found."
    except Exception as e:
        logger.error(f"Error retrieving policies: {str(e)}")
        return "An error occurred while retrieving policies."

async def entrypoint(ctx: JobContext):
    logger.info("Starting entrypoint")
    
    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable()
    async def log_user_data(
        user_booking_number: Annotated[
            str, llm.TypeInfo(description="The user's booking number")
        ],
        user_lastname: Annotated[
            str, llm.TypeInfo(description="The user's last name")
        ],
        user_contact: Annotated[
            str, llm.TypeInfo(description="The user's contact number")
        ],
    ):
        """Logs user data into the Supabase 'logs' table."""
        if not all([user_booking_number, user_lastname, user_contact]):
            logger.warning("Invalid user data provided for logging.")
            return "Invalid data provided. Please provide valid user details."

        try:
            # Insert data into the 'logs' table
            response = supabase.table("logs").insert({
                "booking_number": user_booking_number,
                "user_lastname": user_lastname,
                "contact_number": user_contact,
            }).execute()

            # Check response
            if hasattr(response, "data") and response.data:
                logger.info(f"Data inserted successfully: {response.data}")
                return f"Successfully logged data for user {user_lastname}."
            else:
                logger.error("Data insertion failed or response structure unexpected.")
                return f"Failed to log data for user {user_lastname}."
            
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error logging data: {str(e)}")
            return f"An unexpected error occurred while logging data for user {user_lastname}."

    @fnc_ctx.ai_callable()
    async def retrieve_policies_function(query: str) -> str:
        """Retrieve policy-related data from Supabase."""
        return await retrieve_policies(query)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    chat_ctx = llm.ChatContext()
    chat_ctx.append(
        role="system",
        text=(
            "You are Friday, an intelligent assistant of Anderson Airlines, a commercial airline company. "
            "Your primary interface with users is through voice, so maintain a friendly and conversational tone."
            "Be sure to introduce yourself."
            "Avoid using complex or unpronounceable punctuation to ensure clear communication."
            "Stay professional and supportive."
            "Prioritize speaking in Tagalog."
            "You help users in answering the airline's policy on refundable tickets."
            "Currently, the refund policy is only when the user wants to change their flight or refund their flight before 4 hours of the actual flight. If they want a refund, you will be asking their last name, contact number and booking number."
            "Verify by repeating the user's first name, last name, contact number and booking number."
            "Once you have successfully inserted the user's data, tell the user that a human agent will call in 45-60 minutes."
            "When the user sounds angry, show empathy and understanding, assure that a human agent will reach out to him."
            "When a user asks about Anderson Airlines' policies, call the 'retrieve_policies_function' function to retrieve policy-related data."
            "Do not attempt to answer policy-related questions on your own. Always call the 'retrieve_policies_function' function to get the most accurate and up-to-date information."
            "Do not answer questions that are not related to Anderson Arlines and its policy about refunds."
        ),
    )

    agent = multimodal.MultimodalAgent(
        model=openai.realtime.RealtimeModel(
            voice="alloy",
            temperature=0.6,
            instructions="You are a helpful customer support assistant.",
            turn_detection=openai.realtime.ServerVadOptions(
                threshold=0.6, prefix_padding_ms=200, silence_duration_ms=500
            )
        ),
        fnc_ctx=fnc_ctx,
        chat_ctx=chat_ctx,
    )
    
    @agent.on("agent_speech_committed")
    @agent.on("agent_speech_interrupted")
    def _on_agent_speech_created(msg: llm.ChatMessage):
        max_ctx_len = 100
        chat_ctx = agent.chat_ctx_copy()
        if len(chat_ctx.messages) > max_ctx_len:
            chat_ctx.messages = chat_ctx.messages[-max_ctx_len:]
            asyncio.create_task(agent.set_chat_ctx(chat_ctx))

    agent.start(ctx.room, participant)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )