import os
import time
import asyncio
import logging

from dotenv import load_dotenv
from livekit import api
from livekit.protocol.sip import CreateSIPParticipantRequest

logger = logging.getLogger(__name__)

async def outbound_call(phone_number: str) -> str:
    livekit_api = api.LiveKitAPI()
    logger.info(f"Phone number: {phone_number}")
    sip_trunk_id = os.getenv("LIVEKIT_SIP_TRUNK_ID")
    room_name = os.getenv("LIVEKIT_SIP_ROOM_NAME") 
    if not sip_trunk_id:
        raise ValueError("LIVEKIT_SIP_TRUNK_ID is not set in the environment.")
    if not room_name:
        raise ValueError("LIVEKIT_SIP_ROOM_NAME is not set in the environment.")

    request = CreateSIPParticipantRequest(
        sip_trunk_id=sip_trunk_id,
        sip_call_to=phone_number,
        room_name=room_name,
        participant_identity=f"sip-{phone_number}",
        participant_name="Outbound Caller"
    )
    
    try:
        participant_info = await livekit_api.sip.create_sip_participant(request)
        logger.info(f"SIP participant created: {participant_info}")
    except Exception as e:
        logger.error(f"Error creating SIP participant: {e}")
        raise
    finally:
        await livekit_api.aclose()
    
    return f"Successfully created SIP participant: {participant_info}"
