import logging
import os
from typing import List
from asyncio import Lock

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, deepgram, silero
from langchain_openai.embeddings import OpenAIEmbeddings
from supabase import create_client, Client

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load environment variables
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

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

def enrich_with_rag(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext):
    """
    Enriches the LLM context with relevant information retrieved from the memories table.

    Args:
        agent (VoicePipelineAgent): The voice assistant pipeline agent.
        chat_ctx (llm.ChatContext): The current chat context.
    """
    try:
        user_msg = chat_ctx.messages[-1]
        query = user_msg.content
        #logger.info(f"User query for RAG enrichment: {query}")

        # Retrieve relevant context using embeddings
        embedding_results = retrieve_memories(query)  # Call synchronously

        if embedding_results:
            # Combine results into a single string (e.g., first result or joined results)
            enriched_context = embedding_results[0]  # Use the first result
            #logger.info(f"Enriching context with RAG results: {enriched_context}")

            rag_message = llm.ChatMessage.create(
                text=f"Context:\n{enriched_context}",
                role="assistant",
            )

            # Add enriched context as the assistant's response
            chat_ctx.messages[-1] = rag_message
            chat_ctx.messages.append(user_msg)

        else:
            logger.warning("No relevant context found for RAG enrichment.")

    except Exception as e:
        logger.error(f"Error during RAG enrichment: {e}")



conversation_logs = []
log_lock = Lock()

async def entrypoint(ctx: JobContext):
    logger.info("Agent starting...")
    """Main entry point for the voice assistant."""
    global conversation_logs  # Use a global or external storage for persistence

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are an intelligent assistant designed by Anderson to assist him with his professional work. Your primary interface with users is through voice, so maintain a friendly and conversational tone. Avoid using complex or unpronounceable punctuation to ensure clear communication. Make sure to add more insights on the topic as much as possible. Be a know it all assistant." 
            "Your role is to help Anderson recall details about his work experience, projects, articles, and techniques. Anderson works at Sycip Gorres Velayo (SGV) under Ernst & Young (EY) Philippines in the Financial Services Organization (FSO) Technology Consulting. His supervisor is Christian G. Lauron (CGL), who leads the entire FSO."
            "Stay professional and supportive, tailoring your responses to help Anderson with his tasks and objectives."
            "Prioritize in speaking the Tagalog language."
        ),
    )

    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant {participant.identity}")

    # Initialize the assistant
    assistant = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(
            model="chatgpt-4o-latest",
            temperature=0.2,
            ),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        interrupt_speech_duration=0.5,
        interrupt_min_words=0,
        before_llm_cb=lambda agent, ctx: enrich_with_rag(agent, ctx),
    )

    # Start the assistant
    assistant.start(ctx.room, participant)

    # Log and send events to the conversation logs
    async def log_event(event):
        async with log_lock:  # Lock for thread safety
            conversation_logs.append(event)
            logger.info(f"Event logged: {event}")


    await log_event({"type": "message", "content": "Assistant started"})
    await assistant.say("Hey, what can I assist you with?", allow_interruptions=True)
    await log_event({"type": "assistant_message", "content": "Hey, what can I assist you with?"})

    return app

@app.get("/")
async def read_root():
    return {"message": "Hello World!"}

@app.get("/api/conversation")
async def get_logs():
    async with log_lock:
        logs = list(conversation_logs)  # Return a copy of the logs
    return JSONResponse(content={"logs": logs})


@app.get("/api/conversation-logs")
async def get_conversation_logs():
    return {"logs": "example log"}



if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )