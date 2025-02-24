import aiohttp

async def get_current_time(user_timezone: str = "Asia/Manila") -> str:
    """
    Returns the current time from worldtimeapi.org for the specified timezone.
    Defaults to Asia/Manila (Philippines) if no timezone is provided.
    
    :param user_timezone: A timezone string (e.g. "America/New_York").
    :return: The current datetime as a string or an error message.
    """
    url = f"http://worldtimeapi.org/api/timezone/{user_timezone}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract the "datetime" field from the JSON response.
                    return data.get("datetime", "Time data not available.")
                else:
                    return f"Error: Unable to fetch time data for timezone '{user_timezone}'. HTTP status: {response.status}"
    except Exception as e:
        return f"Error: {e}"