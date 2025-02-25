import os
import aiosmtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")

GMAIL_PW = os.getenv("GMAIL_PW")

async def send_email(recipient: str, subject: str, message: str) -> str:
    """
    Sends an email via Gmail's SMTP server using a dynamic subject.
    
    :param recipient: Email address of the recipient.
    :param subject: The email subject.
    :param message: The content/body of the email.
    :return: A status message indicating success or failure.
    """
    # Gmail SMTP configuration.
    smtp_host = "smtp.gmail.com"
    smtp_port = 587
    username = ""
    password = ""

    # Create the email message.
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = username
    msg["To"] = recipient

    try:
        await aiosmtplib.send(
            msg,
            hostname=smtp_host,
            port=smtp_port,
            start_tls=True,
            username=username,
            password=password,
        )
        return "Email sent successfully."
    except Exception as e:
        return f"Failed to send email: {e}"