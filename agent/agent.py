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


def prewarm(proc: JobProcess):
    """Pre-warm resources like VAD for faster startup."""
    proc.userdata["vad"] = silero.VAD.load()

def retrieve_memories(query: str, table_name: str = "memories", k: int = 10) -> List[str]:
    """
    Retrieves relevant embedding data from the Supabase 'memories' table using hybrid search.

    Args:
        query (str): The input query to search for similar embeddings.
        table_name (str): The name of the Supabase table containing embeddings. Default is 'memories'.
        k (int): The number of top matches to retrieve. Default is 5.

    Returns:
        List[str]: A list of the top matching documents' content.
    """
    try:
        # Initialize OpenAI Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Embed the query
        query_vector = embeddings.embed_query(query)  # Synchronous

        # Perform a Supabase query to find the most similar vectors
        response = supabase.rpc(
            "hybrid_search",
            {
                "query_text": query,
                "query_embedding": query_vector,
                "match_count": k,
            }
        ).execute()  # Execute the RPC call synchronously

        # Log the response for debugging
        #logger.debug(f"Supabase RPC response: {response}")

        # Access the data from the response
        if response.data:  # Check if data is present
            return [result["content"] for result in response.data]
        else:
            logger.warning("No results found in RPC response.")
            return []

    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        return ["I'm sorry, I couldn't retrieve relevant information at this time."]

class ContextEnrichment:
    @staticmethod
    def enrich_with_rag(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        """
        Enriches the LLM context with relevant information retrieved from the memories table.
        """
        try:
            user_msg = chat_ctx.messages[-1]
            query = user_msg.content

            # Retrieve relevant context using embeddings
            embedding_results = retrieve_memories(query)

            if embedding_results:
                enriched_context = embedding_results[0]
                rag_message = llm.ChatMessage.create(
                    text=f"Context:\n{enriched_context}",
                    role="assistant",
                )
                chat_ctx.messages[-1] = rag_message
                chat_ctx.messages.append(user_msg)
            else:
                logger.warning("No relevant context found for RAG enrichment.")

        except Exception as e:
            logger.error(f"Error during RAG enrichment: {e}")
    
class AssistantFnc(llm.FunctionContext):
    """
    The class defines a set of LLM functions that the assistant can execute.
    """

    @llm.ai_callable()
    async def log_user_data(
        self,
        user_reference: Annotated[
            str, llm.TypeInfo(description="The user's unique reference ID")
        ],
        user_firstname: Annotated[
            str, llm.TypeInfo(description="The user's first name")
        ],
        user_lastname: Annotated[
            str, llm.TypeInfo(description="The user's first name")
        ],
    ):
        """Logs user data into the Supabase 'logs' table."""
        # Ensure valid input before attempting to log
        if not user_reference or not user_firstname or not user_lastname:
            logger.warning("Invalid user data provided for logging.")
            return "Invalid data provided. Please provide a valid user_reference and user_firstname."

        try:
            # Insert data into the 'logs' table
            response = supabase.table("logs").insert({
                "user_reference": user_reference,
                "user_firstname": user_firstname,
                "user_lastname": user_lastname,
            }).execute()

            # Check response
            if hasattr(response, "data") and response.data:
                logger.info(f"Data inserted successfully: {response.data}")
                return f"Successfully logged data for user {user_firstname}."
            else:
                logger.error("Data insertion failed or response structure unexpected.")
                return f"Failed to log data for user {user_firstname}."

        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error logging data: {str(e)}")
            return f"An unexpected error occurred while logging data for user {user_firstname}."

    

async def entrypoint(ctx: JobContext):
    logger.info("Starting entrypoint")
    
    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable()
    async def log_user_data(
        user_booking_number: Annotated[
            str, llm.TypeInfo(description="The user's booking number")
        ],
        user_firstname: Annotated[
            str, llm.TypeInfo(description="The user's first name")
        ],
        user_lastname: Annotated[
            str, llm.TypeInfo(description="The user's last name")
        ],
        user_contact: Annotated[
            str, llm.TypeInfo(description="The user's contact number")
        ],
    ):
        """Logs user data into the Supabase 'logs' table."""
        if not all([user_booking_number, user_firstname, user_lastname, user_contact]):
            logger.warning("Invalid user data provided for logging.")
            return "Invalid data provided. Please provide valid user details."

        try:
            # Insert data into the 'logs' table
            response = supabase.table("logs").insert({
                "booking_number": user_booking_number,
                "user_firstname": user_firstname,
                "user_lastname": user_lastname,
                "contact_number": user_contact,
            }).execute()

            # Check response
            if hasattr(response, "data") and response.data:
                logger.info(f"Data inserted successfully: {response.data}")
                return f"Successfully logged data for user {user_firstname}."
            else:
                logger.error("Data insertion failed or response structure unexpected.")
                return f"Failed to log data for user {user_firstname}."
            
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error logging data: {str(e)}")
            return f"An unexpected error occurred while logging data for user {user_firstname}."

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    chat_ctx = llm.ChatContext()
    chat_ctx.append(
        role="system",
        text=(
            "You are Friday, an intelligent assistant of Anderson Airlines, a commercial airline company. "
            "Your primary interface with users is through voice, so maintain a friendly and conversational tone."
            "Avoid using complex or unpronounceable punctuation to ensure clear communication."
            "Make sure to add more insights on the topic as much as possible. Be a know-it-all assistant."
            #"Your role is to help Anderson recall details about his work experience, projects, articles, and techniques. Anderson works at Sycip Gorres Velayo (SGV) under Ernst & Young (EY) Philippines in the Financial Services Organization (FSO) Technology Consulting. His supervisor is Christian G. Lauron (CGL), who leads the entire FSO."
            "Stay professional and supportive."
            #"During the COVID-19 Pandemic, Anderson had a project along with Christian Lauron and Christian Chua on Vaccine Information Management System and Vaccine Certificate (VaxCert)."
            "Prioritize speaking in Tagalog."
            "You help users in answering the airline's policy on refundable tickets."
            "Currently, the refund policy is only when the user wants to change their flight or refund their flight before 4 hours of the actual flight. If they want a refund, you will be asking their first name, last name, contact number and booking number."
            "Verify by repeating the user's first name, last name, contact number and booking number."
            "Once you have successfully inserted the user's data, tell the user that a human agent will call in 45-60 minutes."
            #"You are strictly to provide response related to you (Friday), Anderson, SGV EY, Anderson's Projects, Bangko Sentral ng Pilipinas (BSP) circulars/policy or log the user's reference number, first name and last name. Just say you don't know when you are asked anything not related."
            #"You are not interacting with Anderson but public users."
        ),
    )

    agent = multimodal.MultimodalAgent(
        model=openai.realtime.RealtimeModel(
            modalities=["text"],
            voice="alloy",
            temperature=0.6,
            instructions="You are a helpful assistant.",
            turn_detection=openai.realtime.ServerVadOptions(
                threshold=0.6, prefix_padding_ms=200, silence_duration_ms=500
            ),
            max_output_tokens=1500,
        ),
        fnc_ctx=fnc_ctx,
        chat_ctx=chat_ctx,
    )
    
    @agent.on("agent_speech_committed")
    @agent.on("agent_speech_interrupted")
    def _on_agent_speech_created(msg: llm.ChatMessage):
        max_ctx_len = 10
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