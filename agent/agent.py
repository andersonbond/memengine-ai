import sys
print("Using Python executable:", sys.executable)
import os
import logging
from typing import List, Annotated, Optional
from datetime import datetime
import random
import re
import urllib
import aiohttp
import asyncio
import time
import json
import importlib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
official_openai = importlib.import_module("openai")

from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_openai.embeddings import OpenAIEmbeddings

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero
from livekit.rtc import RemoteParticipant, Track, TrackKind, AudioStream

from tools.embed_memory import embed_and_store, retrieve_memories
from tools.log_user_data import log_user_data as handle_log_user_data
from tools.retrieve_policies import retrieve_policies as handle_retrieve_policies
from tools.outbound_call import outbound_call as handle_outbound_call
from tools.weather import get_weather
from tools.worldtime import get_current_time
from tools.gmail import send_email
from tools.timekeeper import record_time, get_time_records

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Initialize logger
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

class MyAgent(Agent):
    def __init__(self) -> None:
        # Load the instruction prompt
        instruction_prompt_file_path = os.path.join(
            os.path.dirname(__file__), "prompt", "instructions.txt"
        )
        with open(instruction_prompt_file_path, "r") as instruction_prompt_file:
            instruction_prompt = instruction_prompt_file.read()

        # Load the system prompt
        prompt_file_path = os.path.join(
            os.path.dirname(__file__), "prompt", "system_prompt.txt"
        )
        with open(prompt_file_path, "r") as prompt_file:
            system_prompt = prompt_file.read()

        # Add current date and time to the system prompt for context
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        username = "Anderson"
        system_prompt += f"\n\nUsername: {username}\n\nCurrent Date and Time: {current_datetime}"

        # Combine prompts
        full_instructions = f"{system_prompt}\n\n{instruction_prompt}"

        super().__init__(
            instructions=full_instructions
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def embed_memory(
        self,
        context: RunContext,
        user: str,
        memory: str
    ) -> str:
        """Called when the user asks to store a memory when it wants to use it for future reference, or share personal schedule availibility."""
        try:
            logger.info(f"Storing memory for user {user}: {memory}")
            current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Current date and time: {current_datetime}")
            memory_content = f"User: {user}\nDate: {current_datetime}\nMemory: {memory}"
            
            result = await embed_and_store(content=memory_content, user=user)
            logger.info(f"Memory storage result: {result}")
            
            return f"Memory has been stored for {user}: {memory}"
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return "An error occurred while adding memory."

    @function_tool
    async def retrieve_policies(
        self,
        context: RunContext,
        query: str
    ) -> str:
        """Called when the user explains the incident or asks about insurance policies. Policy-related from Anderson Bank and Insurance database."""
        try:
            logger.info(f"Retrieving policies with query: {query}")
            result = await handle_retrieve_policies(query)
            logger.info(f"Policy retrieval result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error occurred in retrieve_policies_function: {e}")
            return "An error occurred while processing your request."

    @function_tool
    async def weather_check(
        self,
        context: RunContext,
        location: str
    ) -> str:
        """Called when the user asks about the weather. This function will return the weather for the given location."""
        try:
            logger.info(f"Checking weather for location: {location}")
            result = await get_weather(location)
            logger.info(f"Weather check result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error occurred in weather_check: {e}")
            return "An error occurred while processing your request."

    @function_tool
    async def time_check(
        self,
        context: RunContext,
        timezone: str = "Asia/Manila"
    ) -> str:
        """Called when the user asks for the current date and time. Returns the current date and time for the given timezone."""
        try:
            logger.info(f"Checking time for timezone: {timezone}")
            result = await get_current_time(timezone)
            logger.info(f"Time check result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error occurred in time_check: {e}")
            return "An error occurred while processing your request."

    @function_tool
    async def get_time_records(
        self,
        context: RunContext,
        username: str,
        date: Optional[str] = None
    ) -> str:
        """Retrieves time records for a user from the timesheet table."""
        try:
            result = await get_time_records(username, date)
            if result["success"]:
                if not result["records"]:
                    return f"No time records found for {username}"
                
                records_text = []
                for record in result["records"]:
                    record_text = (
                        f"Date: {record['date']}, Time: {record['time']}, "
                        f"Type: {record['category']}"
                    )
                    if record['remarks']:
                        record_text += f", Remarks: {record['remarks']}"
                    records_text.append(record_text)
                
                return "\n".join(records_text)
            else:
                return f"Failed to retrieve time records: {result.get('error', 'Unknown error')}"
        except Exception as e:
            logger.error(f"Error in get_time_records_function: {e}")
            return "An error occurred while retrieving time records."

    @function_tool
    async def retrieve_memories(
        self,
        context: RunContext,
        query: str
    ) -> str:
        """Called when the use wants to remember something. Retrieve relevant memories using semantic search."""
        try:
            logger.info(f"Retrieving memories query: {query}")
            result = await retrieve_memories(query=query)
            logger.info(f"Memory retrieval result: {result}")
            
            if result["success"]:
                if not result["memories"]:
                    logger.info("No relevant memories found")
                    return "No relevant memories found."
                
                memories_text = []
                for memory in result["memories"]:
                    memory_text = f"Memory: {memory['content']}"
                    if memory.get('created_at'):
                        memory_text += f"\nCreated: {memory['created_at']}"
                    memories_text.append(memory_text)
                
                formatted_result = "\n\n".join(memories_text)
                logger.info(f"Formatted memories: {formatted_result}")
                return formatted_result
            else:
                error_msg = f"Failed to retrieve memories: {result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                return error_msg
        except Exception as e:
            logger.error(f"Error in retrieve_memories_function: {e}")
            return "An error occurred while retrieving memories."

def prewarm(proc: JobProcess):
    """Pre-warm resources like VAD for faster startup."""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    logger.info("Starting entrypoint")
    
    # Add connection error handling
    @ctx.room.on("connection_state_changed")
    def handle_connection_state(state):
        logger.info(f"Connection state changed to: {state}")
        if state == "disconnected":
            logger.info("Attempting to reconnect...")
            asyncio.create_task(reconnect_agent())

    async def reconnect_agent():
        try:
            logger.info("Reconnecting agent...")
            new_model = await create_realtime_model()
            agent.model = new_model
            logger.info("Successfully reconnected to OpenAI")
        except Exception as e:
            logger.error(f"Failed to reconnect to OpenAI: {e}")
            await asyncio.sleep(3)
            asyncio.create_task(reconnect_agent())

    # Initialize the session with all components
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(
            model="gpt-4o-mini-realtime-preview",
            temperature=0.6
        ),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=openai.TTS(voice="shimmer"),
        turn_detection=silero.VAD.load(),
    )

    # Set up metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with our custom agent
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    # Join the room when agent is ready
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm
        ),
    )