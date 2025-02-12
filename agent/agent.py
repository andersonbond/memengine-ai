import logging
import os
from typing import List, Annotated
from datetime import datetime
import random
import re
import urllib
import aiohttp
import asyncio
import time

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

async def _forward_transcription(
    stt_stream: stt.SpeechStream,
    stt_forwarder: transcription.STTSegmentsForwarder,
):
    """Forward the transcription and log the transcript in the console"""
    async for ev in stt_stream:
        stt_forwarder.update(ev)
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            print(ev.alternatives[0].text, end="")
        elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            print("\n")
            print(" -> ", ev.alternatives[0].text)

async def retrieve_policies(query: str) -> str:
    """Retrieve policy-related data from Supabase using embeddings."""
    logger.info(f"Starting policy retrieval: {query}")
    start_time = datetime.now()
    try:
        await asyncio.sleep(2)
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
            logger.info(f"Policy retrieval completed in {datetime.now() - start_time}")
            return policies
        else:
            logger.info(f"Policy retrieval completed in {datetime.now() - start_time}")
            return "No relevant policies found."
    except Exception as e:
        logger.error(f"Error retrieving policies: {str(e)}")
        return "An error occurred while retrieving policies."

async def entrypoint(ctx: JobContext):
    logger.info("Starting entrypoint")
    stt = STT()
    tasks = []
    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable()
    async def log_user_data(
        user_firstname: Annotated[
            str, llm.TypeInfo(description="The customer's first name")
        ],
        # user_lastname: Annotated[
        #     str, llm.TypeInfo(description="The customer's last name")
        # ],
        user_contact: Annotated[
            str, llm.TypeInfo(description="The customer's contact number")
        ],
        user_plate_number: Annotated[
            str, llm.TypeInfo(description="The customer's motorbike plate number")
        ],
        incident: Annotated[
            str, llm.TypeInfo(description="The customer's incident")
        ],
        evaluation: Annotated[
            str, llm.TypeInfo(description="Friday's evaluation based on the insurance policies")
        ],
    ):
        await asyncio.sleep(2)
        """Logs user data into the database 'logs' table."""
        if not all([user_firstname, user_contact, user_plate_number, incident, evaluation]):
            logger.warning("Invalid user data provided for logging.")
            return "Invalid data provided. Please provide valid user details."
        
        try:
            await asyncio.sleep(2)
            response = await asyncio.to_thread(
                lambda: supabase.table("logs").insert({
                    "user_firstname": user_firstname,
                    "contact_number": user_contact,
                    "plate_number": user_plate_number,
                    "incident": incident,
                    "evaluation": evaluation,
                }).execute()
            )
            
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

    @fnc_ctx.ai_callable()
    async def retrieve_policies_function(query: str) -> str:
        """Retrieve policy-related data from Supabase."""
        
        return await retrieve_policies(query)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()


    prompt_file_path = os.path.join(
        os.path.dirname(__file__), "prompt", "system_prompt.txt"
    )
    # Read the system prompt from the file
    with open(prompt_file_path, "r") as prompt_file:
        system_prompt = prompt_file.read()

    chat_ctx = llm.ChatContext()
    chat_ctx.append(
        role="system",
        text=system_prompt
    )

    agent = multimodal.MultimodalAgent(
        model=openai.realtime.RealtimeModel(
            model="gpt-4o-mini-realtime-preview",
            #model="gpt-4o-realtime-preview",
            voice="shimmer",
            temperature=0.6,
            instructions="You are Friday, a helpful customer support assistant of Anderson Bank and Insurance Company.",
            turn_detection=openai.realtime.ServerVadOptions(
                threshold=0.6, prefix_padding_ms=300, silence_duration_ms=600
            )
        ),
        fnc_ctx=fnc_ctx,
        chat_ctx=chat_ctx,
    )
    
    @agent.on("agent_speech_committed")
    def _on_agent_speech_created(msg: llm.ChatMessage):
        max_ctx_len = 100
        chat_context_copy = agent.chat_ctx_copy()
        if len(chat_context_copy.messages) > max_ctx_len:
            chat_context_copy.messages = chat_context_copy.messages[-max_ctx_len:]
            asyncio.create_task(agent.set_chat_ctx(chat_context_copy))

    agent.start(ctx.room, participant)

    async def transcribe_track(participant: RemoteParticipant, track: Track): 
        """Handles audio track transcription."""
        audio_stream = AudioStream(track)
        stt_forwarder = STTSegmentsForwarder(
            room=ctx.room, participant=participant, track=track 
        )
        stt_stream = stt.stream()
        stt_task = asyncio.create_task(_forward_transcription(stt_stream, stt_forwarder))
        tasks.append(stt_task)

        async for ev in audio_stream:
            stt_stream.push_frame(ev.frame)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: Track, publication, participant: RemoteParticipant):
        """Triggers when a new audio track is subscribed."""
        if track.kind == TrackKind.KIND_AUDIO:
            tasks.append(asyncio.create_task(transcribe_track(participant, track)))


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )