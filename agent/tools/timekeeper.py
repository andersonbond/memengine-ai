import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def record_time(
    username: str,
    category: str,
    remarks: Optional[str] = None
) -> Dict[str, Any]:
    """
    Records time in/out for a user in the timesheet table.
    
    Args:
        username (str): The username of the person recording time
        category (str): Either 'time_in' or 'time_out'
        remarks (Optional[str]): Additional remarks about the time record
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - message (str): A descriptive message about the operation
            - error (Optional[str]): Error message if any
    """
    result = {
        "success": False,
        "message": "",
        "error": None
    }
    
    try:
        # Validate category
        if category not in ['time_in', 'time_out']:
            raise ValueError("Category must be either 'time_in' or 'time_out'")
        
        # Get current date and time in a single call
        now = datetime.now()
        current_date = now.strftime('%Y-%m-%d')
        current_time = now.strftime('%H:%M:%S')
        
        # Prepare data for insertion
        time_record = {
            "username": username,
            "category": category,
            "remarks": remarks,
            "date": current_date,
            "time": current_time
        }
        
        # Insert into Supabase
        response = supabase.table("timesheet").insert(time_record).execute()
        
        if response.data:
            result["success"] = True
            result["message"] = f"Successfully recorded {category} for {username} at {current_time}"
            logger.info(result["message"])
        else:
            result["message"] = f"Failed to record {category} for {username}"
            logger.error(result["message"])
            
    except Exception as e:
        error_msg = f"Error recording time: {str(e)}"
        logger.error(error_msg)
        result["error"] = error_msg
        result["message"] = "An error occurred while recording time"
    
    return result

async def get_time_records(
    username: str,
    date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieves time records for a user from the timesheet table.
    
    Args:
        username (str): The username to get records for
        date (Optional[str]): Specific date to filter records (YYYY-MM-DD format)
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - records (List[Dict]): List of time records
            - error (Optional[str]): Error message if any
    """
    result = {
        "success": False,
        "records": [],
        "error": None
    }
    
    try:
        # Build query
        query = supabase.table("timesheet").select("*").eq("username", username)
        
        # Add date filter if provided
        if date:
            query = query.eq("date", date)
            
        # Order by date and time
        query = query.order("date", desc=True).order("time", desc=True)
        
        # Execute query
        response = query.execute()
        
        if response.data:
            result["success"] = True
            result["records"] = response.data
            logger.info(f"Retrieved {len(response.data)} time records for {username}")
        else:
            logger.info(f"No time records found for {username}")
            
    except Exception as e:
        error_msg = f"Error retrieving time records: {str(e)}"
        logger.error(error_msg)
        result["error"] = error_msg
    
    return result 