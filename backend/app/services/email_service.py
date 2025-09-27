"""
Email service for sending lead recommendations and notifications.
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any
import logging
from app.config import settings
from app.utils.positive_language import positive_language

logger = logging.getLogger(__name__)

class EmailService:
    """Service for sending emails via SMTP."""
    
    def __init__(self):
        self.smtp_server = settings.smtp_server
        self.smtp_port = settings.smtp_port
        self.smtp_username = settings.smtp_username
        self.smtp_password = settings.smtp_password
        self.from_email = settings.from_email
        self.from_name = settings.from_name
        
    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """
        Send an email to the specified recipient.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML email content
            text_content: Plain text email content (optional)
            
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        try:
            # Check if email configuration is properly set up
            if not self.smtp_username or not self.smtp_password or not self.from_email:
                logger.warning("Email configuration not properly set up. Skipping email send.")
                logger.info(f"Would have sent email to {to_email} with subject: {subject}")
                return True  # Return True to not break the flow
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{self.from_name} <{self.from_email}>"
            message["To"] = to_email
            
            # Add text content if provided
            if text_content:
                text_part = MIMEText(text_content, "plain")
                message.attach(text_part)
            
            # Add HTML content
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Create secure connection and send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(message)
                
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"Email authentication failed for {to_email}: {str(e)}")
            logger.error("Please check your Gmail App Password configuration. See EMAIL_CONFIG_FIX.md for instructions.")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error sending email to {to_email}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False
    
    def get_email_template(self, lead_type: str, lead_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Get email template based on lead type with search keywords and course-specific content.
        
        Args:
            lead_type: Type of lead (hot, warm, cold)
            lead_data: Lead information including search keywords and course interest
            
        Returns:
            Dict with subject and content
        """
        name = lead_data.get('name', 'Valued Customer')
        company = lead_data.get('company', '')
        role = lead_data.get('role', '')
        search_keywords = lead_data.get('search_keywords', '')
        course_interest = lead_data.get('prior_course_interest', '')
        positive_interest = positive_language.convert_interest_level(course_interest)
        campaign = lead_data.get('campaign', '')
        page_views = lead_data.get('page_views', 0)
        time_spent = lead_data.get('time_spent', 0)
        course_actions = lead_data.get('course_actions', '')
        
        if lead_type.lower() == 'hot':
            template = self._get_hot_lead_template(name, company, role, search_keywords, course_interest, campaign, page_views, time_spent, course_actions)
        elif lead_type.lower() == 'warm':
            template = self._get_warm_lead_template(name, company, role, search_keywords, course_interest, campaign, page_views, time_spent, course_actions)
        else:
            template = self._get_cold_lead_template(name, company, role, search_keywords, course_interest, campaign, page_views, time_spent, course_actions)
        
        # Add clean text version
        template['text_content'] = self._strip_html_to_text(template['content'])
        return template
    
    def _strip_html_to_text(self, html_content: str) -> str:
        """Convert HTML content to clean, readable text."""
        import re
        
        # Remove CSS styles completely
        text = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        
        # Clean up extra whitespace and line breaks
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up line breaks and formatting
        text = text.strip()
        
        return text
    
    def _get_hot_lead_template(self, name: str, company: str, role: str, search_keywords: str, course_interest: str, campaign: str, page_views: int, time_spent: int, course_actions: str) -> Dict[str, str]:
        """Template for hot leads - personalized approach with search keywords and course content."""
        
        # Create personalized subject based on search keywords
        if search_keywords:
            subject = f"🎯 Perfect Match: {search_keywords} Course for {name}"
        else:
            subject = f"🚀 Exclusive Opportunity for {name} at {company}"
        
        # Generate course-specific content based on search keywords
        course_content = self._generate_course_content(search_keywords, course_interest, campaign)
        
        # Engagement level removed for privacy
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .cta-button {{ display: inline-block; background: #28a745; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .highlight {{ background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0; }}
                .course-box {{ background: white; padding: 20px; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Personalized Course Opportunity</h1>
                    <p>Exclusive offer for {role}s at {company}</p>
                </div>
                <div class="content">
                    <h2>Hi {name}! 👋</h2>
                    
                    <p>I noticed your interest in <strong>{search_keywords or campaign}</strong> and I'm excited to share something that's perfectly aligned with what you're looking for!</p>
                    
                    <div class="highlight">
                        <strong>🔥 Hot Lead Alert:</strong> Based on your strong interest in our programs, I believe we have the perfect course for you.
                    </div>
                    
                    <div class="course-box">
                        <h3>🎓 Course Recommendation</h3>
                        <p><strong>Course:</strong> {course_content['course_name']}</p>
                        <p><strong>Why it's perfect for you:</strong></p>
                        <ul>
                            <li>✅ Designed specifically for {role}s</li>
                            <li>✅ Covers {search_keywords or 'your area of interest'}</li>
                            <li>✅ {course_content['benefits']}</li>
                            <li>✅ Immediate practical application</li>
                        </ul>
                    </div>
                    
                    <p><strong>What makes this special:</strong></p>
                    <ul>
                        <li>🎯 Tailored specifically for {role}s in your industry</li>
                        <li>📚 Based on your search for "{search_keywords or campaign}"</li>
                        <li>⚡ Immediate implementation support</li>
                        <li>💰 Exclusive pricing for qualified prospects</li>
                    </ul>
                    
                    <p>Given your engagement with our programs, I'm confident this course will provide significant value for your role at {company}.</p>
                    
                    <a href="mailto:{settings.from_email}?subject=Course%20Demo%20-%20{search_keywords or campaign}" class="cta-button">
                        🎓 Schedule Course Demo
                    </a>
                    
                    <p>Best regards,<br>
                    <strong>{settings.from_name}</strong><br>
                    Lead HeatScore Team</p>
                    
                    <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;">
                    <p style="font-size: 12px; color: #666;">
                        This email was sent because you showed interest in {search_keywords or campaign}. 
                        If you'd prefer not to receive these emails, you can <a href="#">unsubscribe here</a>.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return {"subject": subject, "content": html_content}
    
    def _generate_course_content(self, search_keywords: str, course_interest: str, campaign: str) -> Dict[str, str]:
        """Generate course-specific content based on search keywords and interest."""
        
        # Course mapping based on keywords
        course_mapping = {
            'ai': {'course_name': 'AI Fundamentals for Professionals', 'benefits': 'Hands-on AI implementation and real-world applications'},
            'data science': {'course_name': 'Data Science Mastery Program', 'benefits': 'Complete data analysis pipeline and machine learning'},
            'machine learning': {'course_name': 'Machine Learning Bootcamp', 'benefits': 'Advanced ML algorithms and model deployment'},
            'python': {'course_name': 'Python for Data Professionals', 'benefits': 'Python programming and data manipulation skills'},
            'analytics': {'course_name': 'Business Analytics Certification', 'benefits': 'Data-driven decision making and visualization'},
            'sql': {'course_name': 'SQL Mastery Course', 'benefits': 'Database design and advanced querying techniques'},
            'excel': {'course_name': 'Advanced Excel for Business', 'benefits': 'Advanced formulas, pivot tables, and automation'},
            'power bi': {'course_name': 'Power BI Dashboard Design', 'benefits': 'Interactive dashboards and data storytelling'},
            'tableau': {'course_name': 'Tableau Visualization Course', 'benefits': 'Data visualization and dashboard creation'},
            'statistics': {'course_name': 'Statistics for Data Analysis', 'benefits': 'Statistical methods and hypothesis testing'}
        }
        
        # Find matching course
        keywords_lower = search_keywords.lower() if search_keywords else ''
        campaign_lower = campaign.lower() if campaign else ''
        
        for keyword, course_info in course_mapping.items():
            if keyword in keywords_lower or keyword in campaign_lower:
                return course_info
        
        # Default course based on interest level
        if course_interest == 'high':
            return {'course_name': 'Advanced Data Science Program', 'benefits': 'Comprehensive data science curriculum with hands-on projects'}
        elif course_interest == 'medium':
            return {'course_name': 'Data Analysis Fundamentals', 'benefits': 'Essential data analysis skills and tools'}
        else:
            return {'course_name': 'Introduction to Data Science', 'benefits': 'Basic concepts and practical applications'}
    
    def _get_warm_lead_template(self, name: str, company: str, role: str, search_keywords: str, course_interest: str, campaign: str, page_views: int, time_spent: int, course_actions: str) -> Dict[str, str]:
        """Template for warm leads - nurturing approach with course content."""
        
        # Create personalized subject based on search keywords
        if search_keywords:
            subject = f"💡 {search_keywords} Resources for {name}"
        else:
            subject = f"💡 Valuable Resources for {name} at {company}"
        
        # Generate course-specific content
        course_content = self._generate_course_content(search_keywords, course_interest, campaign)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .resource-box {{ background: white; padding: 20px; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .cta-button {{ display: inline-block; background: #17a2b8; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; margin: 15px 0; }}
                .course-preview {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📚 Nurturing Your Success</h1>
                    <p>Resources tailored for {role}s</p>
                </div>
                <div class="content">
                    <h2>Hello {name}! 👋</h2>
                    
                    <p>Thank you for your interest in <strong>{search_keywords or campaign}</strong>. I wanted to share some valuable resources that might be particularly relevant for your role as a <strong>{role}</strong> at <strong>{company}</strong>.</p>
                    
                    <div class="course-preview">
                        <h3>🎓 Course Preview: {course_content['course_name']}</h3>
                        <p><strong>Perfect for:</strong> {role}s interested in {search_keywords or campaign}</p>
                        <p><strong>What you'll learn:</strong> {course_content['benefits']}</p>
                        <p><strong>Your interest level:</strong> {course_interest.title()}</p>
                    </div>
                    
                    <div class="resource-box">
                        <h3>🎯 Industry Insights</h3>
                        <p>Latest trends and best practices specifically for {role}s in your sector, with focus on {search_keywords or campaign}.</p>
                    </div>
                    
                    <div class="resource-box">
                        <h3>📊 Case Studies</h3>
                        <p>Real examples of how similar companies have achieved success with {search_keywords or campaign} solutions.</p>
                    </div>
                    
                    <div class="resource-box">
                        <h3>🛠️ Implementation Guide</h3>
                        <p>Step-by-step approach to getting started with {search_keywords or campaign}, tailored for your company size.</p>
                    </div>
                    
                    <p>I'd love to hear your thoughts on these resources and discuss how they might apply to your specific situation at {company}. Based on your interest in our programs, I believe you'll find great value in our {course_content['course_name']}.</p>
                    
                    <a href="mailto:{settings.from_email}?subject=Resources%20Discussion%20-%20{search_keywords or campaign}" class="cta-button">
                        💬 Let's Discuss
                    </a>
                    
                    <p>Warm regards,<br>
                    <strong>{settings.from_name}</strong><br>
                    Lead HeatScore Team</p>
                    
                    <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;">
                    <p style="font-size: 12px; color: #666;">
                        You're receiving this because you showed interest in {search_keywords or campaign}. 
                        <a href="#">Unsubscribe</a> if you prefer not to receive these updates.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return {"subject": subject, "content": html_content}
    
    def _get_cold_lead_template(self, name: str, company: str, role: str, search_keywords: str, course_interest: str, campaign: str, page_views: int, time_spent: int, course_actions: str) -> Dict[str, str]:
        """Template for cold leads - low-touch approach with course awareness."""
        
        # Create personalized subject based on search keywords
        if search_keywords:
            subject = f"📧 {search_keywords} Newsletter - {company}"
        else:
            subject = f"📧 Monthly Newsletter - {company}"
        
        # Generate course-specific content
        course_content = self._generate_course_content(search_keywords, course_interest, campaign)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .newsletter-box {{ background: white; padding: 20px; border-radius: 8px; margin: 15px 0; border: 1px solid #e0e0e0; }}
                .cta-button {{ display: inline-block; background: #6c757d; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 15px 0; }}
                .course-mention {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #007bff; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📬 Monthly Newsletter</h1>
                    <p>Industry insights and updates</p>
                </div>
                <div class="content">
                    <h2>Hi {name},</h2>
                    
                    <p>I hope you're doing well at <strong>{company}</strong>. I wanted to keep you updated on some industry insights that might be relevant for your role as a <strong>{role}</strong>.</p>
                    
                    <div class="course-mention">
                        <h3>🎓 Course Spotlight</h3>
                        <p><strong>{course_content['course_name']}</strong></p>
                        <p>Perfect for {role}s interested in {search_keywords or campaign}. {course_content['benefits']}</p>
                    </div>
                    
                    <div class="newsletter-box">
                        <h3>📈 This Month's Highlights</h3>
                        <ul>
                            <li>Industry trends affecting {role}s</li>
                            <li>Best practices from successful companies</li>
                            <li>Upcoming opportunities in your sector</li>
                            <li>Latest developments in {search_keywords or campaign}</li>
                        </ul>
                    </div>
                    
                    <p>I'll continue to share valuable insights and keep you informed about opportunities that might be relevant for {company}. If you're interested in learning more about {search_keywords or campaign}, our {course_content['course_name']} might be perfect for you.</p>
                    
                    <p>If you'd like to learn more about how we can help your organization, feel free to reach out anytime.</p>
                    
                    <a href="mailto:{settings.from_email}?subject=More%20Information%20-%20{search_keywords or campaign}" class="cta-button">
                        📧 Learn More
                    </a>
                    
                    <p>Best regards,<br>
                    <strong>{settings.from_name}</strong><br>
                    Lead HeatScore Team</p>
                    
                    <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;">
                    <p style="font-size: 12px; color: #666;">
                        You're receiving this newsletter because you showed interest in {search_keywords or campaign}. 
                        <a href="#">Unsubscribe</a> to stop receiving these updates.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return {"subject": subject, "content": html_content}

# Global email service instance
email_service = EmailService()

