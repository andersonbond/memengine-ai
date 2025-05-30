import datetime
import pytz
from typing import Optional

async def get_current_time(timezone: str = "Asia/Manila") -> str:
    """
    Get the current time for a given timezone.
    
    Args:
        timezone (str): The timezone to get the time for (default: Asia/Manila)
        
    Returns:
        str: Formatted string with current date and time
    """
    try:
        # Get the timezone
        tz = pytz.timezone(timezone)
        
        # Get current time in the specified timezone
        current_time = datetime.datetime.now(tz)
        
        # Format the time string
        time_str = current_time.strftime("%B %d, %Y %I:%M %p")
        
        return f"The current time in {timezone} is {time_str}"
    except Exception as e:
        return f"Error getting time: {str(e)}"