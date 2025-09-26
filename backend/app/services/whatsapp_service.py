import asyncio
import aiohttp
from typing import Dict, Any, Optional
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class WhatsAppService:
    """Service for sending WhatsApp messages via WhatsApp Business API."""
    
    def __init__(self):
        self.enabled = bool(settings.whatsapp_business_token)
        self.bot_token = settings.whatsapp_business_token
        self.api_url = f"https://graph.facebook.com/v18.0/{settings.whatsapp_phone_number_id}/messages"
        self.timeout = settings.whatsapp_message_timeout
        
    async def send_message(self, phone_number: str, message: str) -> Dict[str, Any]:
        """Send a WhatsApp message to a phone number."""
        if not self.enabled:
            logger.warning("WhatsApp service is disabled or no token provided")
            return {"success": False, "error": "WhatsApp service disabled"}
        
        # Clean phone number (remove +, spaces, dashes)
        clean_phone = phone_number.replace("+", "").replace(" ", "").replace("-", "")
        
        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messaging_product": "whatsapp",
            "to": clean_phone,
            "type": "text",
            "text": {
                "body": message
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"WhatsApp message sent successfully to {phone_number}")
                        return {
                            "success": True,
                            "message": f"WhatsApp message sent to {phone_number}",
                            "message_id": result.get("messages", [{}])[0].get("id"),
                            "phone_number": phone_number
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to send WhatsApp message: {response.status} - {error_text}")
                        return {
                            "success": False,
                            "error": f"Failed to send message: {response.status} - {error_text}",
                            "phone_number": phone_number
                        }
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {e}")
            return {
                "success": False,
                "error": f"Error sending message: {str(e)}",
                "phone_number": phone_number
            }
    
    def craft_whatsapp_message(self, lead_data: Dict[str, Any], policy_rules: Dict[str, Any]) -> str:
        """Craft a WhatsApp message based on lead data and policy rules."""
        name = lead_data.get("name", "there")
        role = lead_data.get("role", "Professional")
        campaign = lead_data.get("campaign", "our course")
        page_views = lead_data.get("page_views", 0)
        time_spent = lead_data.get("time_spent", 0)
        search_keywords = lead_data.get("search_keywords", "")
        
        # Get tone rules
        tone = policy_rules.get("tone", "professional")
        urgency = policy_rules.get("urgency_level", "medium")
        
        # Craft message based on urgency and personalization
        if urgency == "high":
            message = f"Hi {name}! 🔥\n\n"
            message += f"I noticed your strong interest in {campaign} - {page_views} page views and {time_spent}s exploring!\n\n"
            if search_keywords:
                message += f"Your interest in {search_keywords} shows you're serious about advancing your skills.\n\n"
            message += f"As a {role}, you're perfect for our program. Ready to take the next step?\n\n"
            message += "Let's schedule a call to discuss how this can accelerate your career! 🚀"
        else:
            message = f"Hi {name}! 👋\n\n"
            message += f"Thanks for your interest in {campaign}. I noticed you spent time exploring our content.\n\n"
            if search_keywords:
                message += f"Since you're interested in {search_keywords}, I thought you'd like to know about our upcoming program.\n\n"
            message += f"As a {role}, this could be a great opportunity for you.\n\n"
            message += "Would you like to learn more? Reply 'YES' for details! 📚"
        
        return message
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test WhatsApp Business API connection."""
        if not self.enabled:
            return {"success": False, "error": "WhatsApp service disabled"}
        
        try:
            # Test with a simple API call to verify token
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://graph.facebook.com/v18.0/{settings.whatsapp_phone_number_id}",
                    headers=headers,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        logger.info("WhatsApp Business API connection successful")
                        return {
                            "success": True,
                            "message": "WhatsApp Business API connection successful",
                            "phone_number_id": settings.whatsapp_phone_number_id
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"WhatsApp API test failed: {response.status} - {error_text}")
                        return {
                            "success": False,
                            "error": f"API test failed: {response.status} - {error_text}"
                        }
        except Exception as e:
            logger.error(f"Error testing WhatsApp connection: {e}")
            return {
                "success": False,
                "error": f"Connection test failed: {str(e)}"
            }


# Global service instance
whatsapp_service = WhatsAppService()
