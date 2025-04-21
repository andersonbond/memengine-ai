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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
official_openai = importlib.import_module("openai")


from dotenv import load_dotenv

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    multimodal,
    stt,
    transcription
)
from livekit.plugins.deepgram import STT
from livekit.agents.pipeline import AgentCallContext, VoicePipelineAgent
from livekit.agents.multimodal import MultimodalAgent
from livekit.rtc import RemoteParticipant, Track, TrackKind, AudioStream

from livekit.plugins import openai, deepgram, silero
from langchain_openai.embeddings import OpenAIEmbeddings
from supabase import create_client, Client

from tools.embed_memory import embed_and_store
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

def prewarm(proc: JobProcess):
    """Pre-warm resources like VAD for faster startup."""
    proc.userdata["vad"] = silero.VAD.load()

async def _forward_transcription(
    stt_stream: stt.SpeechStream,
    stt_forwarder: transcription.STTSegmentsForwarder,
):
    """Forward the transcription and log the transcript in the console."""
    async for ev in stt_stream:
        stt_forwarder.update(ev)
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            print(ev.alternatives[0].text, end="")
        elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            print("\n")
            print(" -> ", ev.alternatives[0].text)

async def entrypoint(ctx: JobContext):
    logger.info("Starting entrypoint")
    
    stt_instance = STT(model="nova-3")
    tasks = []  # To keep track of running tasks

    # FUNCTION CALLING: Register AI callable functions.
    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable()
    async def embed_memory(
        user: str,
        memory: str
    ) -> str:
        """Called when the user asks to store a memory when it wants to use it for future reference, or share personal schedule availibility."""
        try:
            # Get the current date and time
            current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Current date and time: {current_datetime}")
            # Create a memory string in a structured format
            memory_content = f"User: {user}\nDate: {current_datetime}\nMemory: {memory}"
            
            # Use the embed_and_store function with the new implementation
            await embed_and_store(content=memory_content, user=user)
            
            return f"Memory has been stored for {user}: {memory}"
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return "An error occurred while adding memory."

    @fnc_ctx.ai_callable()
    async def log_user_data_function(
        user_firstname: str,
        user_contact: str,
        user_plate_number: str,
        incident: str,
        evaluation: str
    ) -> str:
        """Logs customer data into the database."""
        try:
            return await handle_log_user_data(
                user_firstname,
                user_contact,
                user_plate_number,
                incident,
                evaluation
            )
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            return "An error occurred while processing your request."

    @fnc_ctx.ai_callable()
    async def retrieve_policies_function(query: str) -> str:
        """Called when the user explains the incident or asks about the policies. Policy-related data from Anderson Bank and Insurance database."""
        try:
            return await handle_retrieve_policies(query)
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            return "An error occurred while processing your request."

    @fnc_ctx.ai_callable()
    async def outbound_call_function(
        phone_number: Annotated[
            str, llm.TypeInfo(description="Phone number to call: +639477886466")
        ]
    ) -> str:
        """Called when the customer would like to be transferred to a real human agent. This function will add another participant to the call."""
        try:
            call_sid = await handle_outbound_call("09477886466")
            return f"Outbound call initiated with SID: {call_sid}"
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            return "An error occurred while processing your request."
    
    @fnc_ctx.ai_callable()
    async def weather_check(location: Annotated[
            str, llm.TypeInfo(description="The location to get the weather for")
        ],) -> str:
        """Called when the user asks about the weather. This function will return the weather for the given location."""
        try:
            return await get_weather(location)
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            return "An error occurred while processing your request."

    @fnc_ctx.ai_callable()
    async def time_check(timezone: str = "Asia/Manila") -> str:
        """
        Called when the user asks for the current date and time. Returns the current date and time for the given timezone.
        Defaults to Philippines time (Asia/Manila) if no timezone is provided.
        """
        try:
            return await get_current_time(timezone)
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            return "An error occurred while processing your request."
    
    @fnc_ctx.ai_callable()
    async def send_via_gmail(message: str, email_address: str) -> str:
        """
        Called when the user asks to send an email. Sends an email using the analyzed message content provided by the voice AI.
        
        :param message: The analyzed message content.
        :param email_address: The recipient's email address.
        :return: A confirmation message indicating the email status.
        """
        try:
            subject = "No Subject"

            send_result = await send_email(email_address, subject, message)
            
            return f"Email sent to {email_address}. {send_result}"
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            return "An error occurred while processing your request."
        
    @fnc_ctx.ai_callable()
    async def record_time_function(
        username: str,
        category: str,
        remarks: Optional[str] = None
    ) -> str:
        """
        Records time in/out for a user in the timesheet table.
        
        Args:
            username (str): The username of the person recording time
            category (str): Either 'time_in' or 'time_out'
            remarks (Optional[str]): Additional remarks about the time record
            
        Returns:
            str: A message indicating the success or failure of the operation
        """
        try:
            result = await record_time(username, category, remarks)
            if result["success"]:
                return result["message"]
            else:
                return f"Failed to record time: {result.get('error', 'Unknown error')}"
        except Exception as e:
            logger.error(f"Error in record_time_function: {e}")
            return "An error occurred while recording time."

    @fnc_ctx.ai_callable()
    async def get_time_records_function(
        username: str,
        date: Optional[str] = None
    ) -> str:
        """
        Retrieves time records for a user from the timesheet table.
        
        Args:
            username (str): The username to get records for
            date (Optional[str]): Specific date to filter records (YYYY-MM-DD format)
            
        Returns:
            str: A formatted message containing the time records or an error message
        """
        try:
            result = await get_time_records(username, date)
            if result["success"]:
                if not result["records"]:
                    return f"No time records found for {username}"
                
                # Format the records
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

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    # Load the instruction prompt
    instruction_prompt_file_path = os.path.join(
        os.path.dirname(__file__), "prompt", "instructions.txt"
    )
   
    with open(instruction_prompt_file_path, "r") as instruction_prompt_file:
        instruction_prompt = instruction_prompt_file.read()
        #print("Instruction: ", instruction_prompt)
    # Load the system prompt
    prompt_file_path = os.path.join(
        os.path.dirname(__file__), "prompt", "system_prompt.txt"
    )
    with open(prompt_file_path, "r") as prompt_file:
        system_prompt = prompt_file.read()

     # Add current date and time to the system prompt for context.
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    username = "Anderson"
    system_prompt += f"\n\nUsername: {username}\n\nCurrent Date and Time: {current_datetime}"

    chat_ctx = llm.ChatContext()
    chat_ctx.append(role="system", text=system_prompt)

    # Initialize the OpenAI realtime model with retry logic
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def create_realtime_model():
        return openai.realtime.RealtimeModel(
            model="gpt-4o-mini-realtime-preview",
            voice="shimmer",
            temperature=0.6,
            instructions=(instruction_prompt),
            turn_detection=openai.realtime.ServerVadOptions(
                threshold=0.6, prefix_padding_ms=300, silence_duration_ms=600
            )
        )

    try:
        realtime_model = await create_realtime_model()
    except Exception as e:
        logger.error(f"Failed to create OpenAI realtime model after retries: {e}")
        raise

    agent = multimodal.MultimodalAgent(
        model=realtime_model,
        fnc_ctx=fnc_ctx,
        chat_ctx=chat_ctx,
    )

    # Add error handling for agent events
    @agent.on("error")
    def handle_agent_error(error: Exception):
        logger.error(f"Agent error occurred: {error}")
        if "OpenAI S2S connection closed unexpectedly" in str(error):
            logger.info("Attempting to reconnect to OpenAI...")
            asyncio.create_task(reconnect_agent())

    async def reconnect_agent():
        try:
            logger.info("Reconnecting agent...")
            new_model = await create_realtime_model()
            agent.model = new_model
            logger.info("Successfully reconnected to OpenAI")
        except Exception as e:
            logger.error(f"Failed to reconnect to OpenAI: {e}")

    @agent.on("agent_speech_committed")
    def _on_agent_speech_created(msg: llm.ChatMessage):
        max_ctx_len = 100
        chat_context_copy = agent.chat_ctx_copy()
        if len(chat_context_copy.messages) > max_ctx_len:
            # Keep only the last max_ctx_len messages.
            chat_context_copy.messages = chat_context_copy.messages[-max_ctx_len:]
            # Create a task to update the agent's chat context and add it to our task list.
            task = asyncio.create_task(agent.set_chat_ctx(chat_context_copy))
            tasks.append(task)

    agent.start(ctx.room, participant)

    async def transcribe_track(participant: RemoteParticipant, track: Track): 
        """Handles audio track transcription."""
        audio_stream = AudioStream(track, noise_suppression=True)
        stt_forwarder = transcription.STTSegmentsForwarder(
            room=ctx.room, participant=participant, track=track 
        )
        stt_stream = stt_instance.stream()
        # Create a task for forwarding transcription.
        stt_task = asyncio.create_task(_forward_transcription(stt_stream, stt_forwarder))
        tasks.append(stt_task)

        async for ev in audio_stream:
            stt_stream.push_frame(ev.frame)
    
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: Track, publication, participant: RemoteParticipant):
        """Triggers when a new audio track is subscribed."""
        if track.kind == TrackKind.KIND_AUDIO:
            task = asyncio.create_task(transcribe_track(participant, track))
            tasks.append(task)

    # Keep the entrypoint running indefinitely.
    try:
        await asyncio.Future()  # This future never completes, keeping the loop alive.
    except asyncio.CancelledError:
        logger.info("Entrypoint cancelled. Cleaning up tasks.")
    finally:
        # Cancel all pending tasks and wait for them to finish.
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All tasks have been cleaned up.")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm
        ),
    )