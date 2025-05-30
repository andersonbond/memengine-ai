import os
import logging
from typing import Annotated
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def log_user_data(
    user_firstname: Annotated[str, "The customer's first name"],
    user_contact: Annotated[str, "The customer's contact number"],
    user_plate_number: Annotated[str, "The customer's plate number"],
    incident: Annotated[str, "Description of the incident"],
    evaluation: Annotated[str, "Evaluation of the incident"]
) -> str:
    """Logs customer data into the database."""
    try:
        # Get current timestamp
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Insert data into the 'user_data' table
        response = supabase.table("user_data").insert({
            "user_firstname": user_firstname,
            "user_contact": user_contact,
            "user_plate_number": user_plate_number,
            "incident": incident,
            "evaluation": evaluation,
            "created_at": current_datetime
        }).execute()
        
        if response.data:
            logger.info(f"Successfully logged data for user: {user_firstname}")
            return f"Data has been logged for {user_firstname}"
        else:
            logger.error("Failed to log user data: No data in response")
            return "Failed to log user data"
            
    except Exception as e:
        logger.error(f"Error logging user data: {e}")
        return "An error occurred while logging user data"