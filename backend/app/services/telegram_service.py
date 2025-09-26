import requests
import asyncio
from typing import Dict, Any, Optional, List
from app.config import settings
from app.models.schemas import LeadInput, HeatScore
from app.services.policy_service import policy_service
from app.services.persona_service import persona_service
from app.utils.logging import get_logger

logger = get_logger(__name__)


class TelegramService:
    """Service for sending messages via Telegram Bot API."""
    
    def __init__(self):
        self.bot_token = settings.telegram_bot_token
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.enabled = settings.enable_telegram_messages and bool(self.bot_token)
    
    async def send_message(self, chat_id: str, message: str, parse_mode: str = "HTML") -> Dict[str, Any]:
        """Send message via Telegram Bot API."""
        if not self.enabled:
            logger.warning("Telegram service is disabled or no bot token provided")
            return {"success": False, "error": "Telegram service disabled"}
        
        try:
            url = f"{self.api_url}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            
            # Use asyncio to make the request non-blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(url, json=data, timeout=settings.telegram_message_timeout)
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Telegram message sent successfully to chat {chat_id}")
                return {
                    "success": True,
                    "message_id": result.get("result", {}).get("message_id"),
                    "chat_id": chat_id
                }
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "details": response.text
                }
                
        except requests.exceptions.Timeout:
            logger.error("Telegram API timeout")
            return {"success": False, "error": "API timeout"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Telegram API request error: {e}")
            return {"success": False, "error": f"Request error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def craft_telegram_message(self, lead_data: LeadInput, heat_score: HeatScore) -> str:
        """Craft Telegram message using policy and persona."""
        # Get policy rules
        tone_rules = policy_service.get_tone_rules(heat_score)
        cta_rules = policy_service.get_cta_rules(heat_score)
        
        # Get persona content
        persona = persona_service.get_persona_content(lead_data.role)
        messaging_focus = persona.get("messaging_focus", "")
        value_props = persona.get("value_propositions", [])
        
        # Extract tone and urgency
        tone = tone_rules.get("tone", "professional")
        urgency = cta_rules.get("urgency", "moderate")
        primary_cta = cta_rules.get("primary_cta", "learn_more")
        
        # Craft message based on heat score
        if heat_score == HeatScore.HOT:
            message = self._craft_hot_telegram_message(lead_data, tone, urgency, primary_cta, messaging_focus, value_props)
        elif heat_score == HeatScore.WARM:
            message = self._craft_warm_telegram_message(lead_data, tone, urgency, primary_cta, messaging_focus, value_props)
        else:  # COLD
            message = self._craft_cold_telegram_message(lead_data, tone, urgency, primary_cta, messaging_focus, value_props)
        
        return message
    
    def _craft_hot_telegram_message(self, lead_data: LeadInput, tone: str, urgency: str, cta: str, messaging_focus: str, value_props: List[str]) -> str:
        """Craft urgent Telegram message for hot leads."""
        # Dynamic pricing based on campaign/role
        base_price = 5999
        discount_price = 3999
        discount_percent = int(((base_price - discount_price) / base_price) * 100)
        
        # Personalized course name
        course_name = lead_data.campaign.replace('_', ' ').title()
        
        # Personalized benefits based on role
        role_benefits = {
            'manager': 'Leadership skills & team management',
            'engineer': 'Technical expertise & coding mastery',
            'director': 'Strategic thinking & executive skills',
            'analyst': 'Data analysis & business intelligence',
            'consultant': 'Client management & problem-solving'
        }
        
        primary_benefit = role_benefits.get(lead_data.role.lower(), 'Professional development')
        
        message = f"""👋 Hi {lead_data.name}!

🎯 I noticed your strong interest in our {course_name} program!

📊 Based on your profile as a {lead_data.role}, I believe this could be perfect for your career growth.

🔥 EXCLUSIVE OFFER FOR YOU:
• 🎓 Complete {course_name} certification
• 👨‍💼 Career guidance & placement support  
• 💡 {primary_benefit}
• 🤝 Expert mentorship program
• 📊 Real-world case studies
• 🏆 Industry-recognized certificate

💰 SPECIAL DISCOUNT:
~~₹{base_price:,}~~ **₹{discount_price:,}** - Offer available only for 3 days! ⏰
Enroll today and get {discount_percent}% OFF your chosen course – offer valid till midnight!

📞 Contact Us:
• 📱 Phone: +1-800-LEADHEAT
• 📧 Email: support@leadheatscore.com
• 🌐 Website: www.leadheatscore.com

🚀 Ready to advance your career? 

Reply 'YES' to get started or 'INFO' for more details!

Best regards,
LeadHeatScore Team"""
        
        return message
    
    def _craft_warm_telegram_message(self, lead_data: LeadInput, tone: str, urgency: str, cta: str, messaging_focus: str, value_props: List[str]) -> str:
        """Craft nurturing Telegram message for warm leads."""
        # Personalized course name
        course_name = lead_data.campaign.replace('_', ' ').title()
        
        # Dynamic webinar name based on campaign
        webinar_name = f"{course_name} Masterclass"
        
        # Personalized learning outcomes based on role
        role_outcomes = {
            'manager': 'Team leadership & project management',
            'engineer': 'Advanced coding & system design',
            'director': 'Strategic planning & executive decision-making',
            'analyst': 'Data visualization & business insights',
            'consultant': 'Client relations & solution design'
        }
        
        primary_outcome = role_outcomes.get(lead_data.role.lower(), 'Professional skills development')
        
        message = f"""👋 <b>Hi {lead_data.name}!</b>

📚 Thanks for your interest in our <b>{course_name}</b> program.

🎯 As a <b>{lead_data.role}</b>, this could be a great opportunity for your career development.

🎓 <b>FREE WEBINAR INVITATION:</b>
Join our FREE {webinar_name} this weekend and unlock your learning journey 🚀

✨ <b>What You'll Learn:</b>
• 🎓 Industry-recognized certification
• 📈 Career advancement support
• 💡 {primary_outcome}
• 🤝 Expert mentorship
• 📊 Real-world case studies
• 🏆 Professional portfolio building

📞 <b>Contact Us:</b>
• 📱 Phone: +1-800-LEADHEAT
• 📧 Email: support@leadheatscore.com
• 🌐 Website: www.leadheatscore.com

💬 Want to learn more? 

Reply <b>'WEBINAR'</b> to join or <b>'INFO'</b> for details!

Best regards,
<b>LeadHeatScore Team</b>"""
        
        return message
    
    def _craft_cold_telegram_message(self, lead_data: LeadInput, tone: str, urgency: str, cta: str, messaging_focus: str, value_props: List[str]) -> str:
        """Craft educational Telegram message for cold leads."""
        # Personalized course name
        course_name = lead_data.campaign.replace('_', ' ').title()
        
        # Dynamic success story based on role
        role_success_stories = {
            'manager': '5,000+ managers advanced their careers',
            'engineer': '8,000+ engineers landed dream jobs',
            'director': '2,000+ directors got promoted',
            'analyst': '6,000+ analysts became data experts',
            'consultant': '3,000+ consultants doubled their income'
        }
        
        success_story = role_success_stories.get(lead_data.role.lower(), '10,000+ professionals started their journey')
        
        # Personalized resources based on role
        role_resources = {
            'manager': 'Leadership frameworks & team management guides',
            'engineer': 'Coding tutorials & technical documentation',
            'director': 'Strategic planning templates & executive insights',
            'analyst': 'Data analysis tools & business intelligence guides',
            'consultant': 'Client management strategies & solution frameworks'
        }
        
        primary_resource = role_resources.get(lead_data.role.lower(), 'Industry insights and best practices')
        
        message = f"""📰 <b>Educational Content for {lead_data.name}</b>

Hi {lead_data.name}! 👋

I hope you're doing well. I wanted to share some valuable resources that might be relevant for your role as a <b>{lead_data.role}</b>.

{messaging_focus}

🌟 <b>SUCCESS STORY:</b>
Did you know {success_story} with our free resources? Start your journey today ✨

<b>Free Resources Available:</b>
• {primary_resource}
• Best practices and tips
• Educational content

<b>Stay Updated:</b>
📧 Subscribe to our newsletter
📱 Follow us for daily insights
💡 Access free resources

No pressure - just valuable content when you're ready!

Best regards,
Content Team

<i>Reply with 'NEWS' to subscribe</i>"""
        
        return message
    
    async def send_message_to_phone(self, phone_number: str, message: str, parse_mode: str = "HTML") -> Dict[str, Any]:
        """Send message via Telegram using phone number (requires user to start chat with bot first)."""
        if not self.enabled:
            logger.warning("Telegram service is disabled or no bot token provided")
            return {"success": False, "error": "Telegram service disabled"}
        
        logger.info(f"Attempting to send Telegram message to phone: {phone_number}")
        
        # For now, we'll try to send to a test chat_id
        # In production, you'd need to map phone numbers to chat_ids
        # For testing, you can use your own chat_id
        
        # Get your chat_id by sending a message to @userinfobot on Telegram
        # Or use a test chat_id if you have one
        test_chat_id = "7858603752"  # Your actual chat_id
        
        # Skip demo mode since we have a real chat_id
        # if test_chat_id == "YOUR_CHAT_ID_HERE":
        #     # Fallback to demo mode if no chat_id is configured
        #     return {
        #         "success": True,
        #         "message": f"Telegram message prepared for {phone_number}",
        #         "note": "Configure chat_id in telegram_service.py to enable real sending",
        #         "phone_number": phone_number,
        #         "message_preview": message[:100] + "..." if len(message) > 100 else message
        #     }
        
        # Actually send the message via Telegram API
        try:
            result = await self.send_message(test_chat_id, message, parse_mode)
            if result["success"]:
                logger.info(f"Telegram message sent successfully to {phone_number}")
                return {
                    "success": True,
                    "message": f"Telegram message sent to {phone_number}",
                    "phone_number": phone_number,
                    "message_id": result.get("message_id"),
                    "chat_id": test_chat_id
                }
            else:
                logger.error(f"Failed to send Telegram message: {result.get('error')}")
                return {
                    "success": False,
                    "error": f"Failed to send message: {result.get('error')}",
                    "phone_number": phone_number
                }
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return {
                "success": False,
                "error": f"Error sending message: {str(e)}",
                "phone_number": phone_number
            }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Telegram bot connection."""
        if not self.enabled:
            return {"success": False, "error": "Telegram service disabled"}
        
        try:
            url = f"{self.api_url}/getMe"
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.get(url, timeout=10)
            )
            
            if response.status_code == 200:
                bot_info = response.json().get("result", {})
                logger.info(f"Telegram bot connection successful: {bot_info.get('first_name', 'Unknown')}")
                return {
                    "success": True,
                    "bot_info": bot_info
                }
            else:
                logger.error(f"Telegram bot test failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Telegram bot test error: {e}")
            return {"success": False, "error": str(e)}


# Global service instance
telegram_service = TelegramService()
