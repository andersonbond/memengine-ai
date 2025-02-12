import os
import asyncio
from typing import List, Annotated
import logging

logger = logging.getLogger("voice-agent")

from livekit.agents import (
    llm
)

from supabase import create_client, Client

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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