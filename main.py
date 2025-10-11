

import os
import logging
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import uuid
import sqlite3
from dataclasses import dataclass
import re

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import OutputParserException
import openai

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Unified Academic Chatbot API",
    description="Friend-like academic chatbot that's conversational and context-aware",
    version="1.0.0"
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class CollegeRecommendation(BaseModel):
    """College recommendation model"""
    id: str
    name: str
    location: str
    type: str
    courses_offered: str
    website: str
    admission_process: str
    approximate_fees: str
    notable_features: str
    source: str

class ChatResponse(BaseModel):
    response: str
    is_recommendation: bool
    timestamp: str
    conversation_title: Optional[str] = None
    recommendations: Optional[List[CollegeRecommendation]] = []

# Models
class UserPreferences(BaseModel):
    """User preferences extracted from conversation"""
    location: Optional[str] = Field(None, description="Preferred city or state for college")
    state: Optional[str] = Field(None, description="Preferred state for college")
    course_type: Optional[str] = Field(None, description="Type of course like Engineering, Medicine, Arts, Commerce, etc.")
    college_type: Optional[str] = Field(None, description="Government, Private, or Deemed university")
    level: Optional[str] = Field(None, description="UG (Undergraduate) or PG (Postgraduate)")
    budget_range: Optional[str] = Field(None, description="Budget preference like low, medium, high")
    specific_course: Optional[str] = Field(None, description="Specific course like BTech, MBA, MBBS, etc.")
    specific_institution_type: Optional[str] = Field(None, description="Specific institution type like IIT, NIT, IIIT, AIIMS, etc.")

@dataclass
class College:
    college_id: str
    name: str
    type: str
    affiliation: str
    location: str
    website: str
    contact: str
    email: str
    courses: str
    scholarship: str
    admission_process: str

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                message_type TEXT,
                content TEXT,
                is_recommendation BOOLEAN DEFAULT FALSE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preferences (
                chat_id TEXT PRIMARY KEY,
                preferences TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_titles (
                chat_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_message(self, chat_id: str, message_type: str, content: str, is_recommendation: bool = False):
        """Save a message"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO messages (chat_id, message_type, content, is_recommendation) VALUES (?, ?, ?, ?)',
            (chat_id, message_type, content, is_recommendation)
        )
        conn.commit()
        conn.close()
    
    def get_chat_messages(self, chat_id: str) -> List[Dict]:
        """Get messages for a chat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT message_type, content, timestamp, is_recommendation
            FROM messages 
            WHERE chat_id = ? 
            ORDER BY timestamp
        ''', (chat_id,))
        messages = cursor.fetchall()
        conn.close()
        
        return [
            {
                'type': msg[0],
                'content': msg[1],
                'timestamp': msg[2],
                'is_recommendation': msg[3]
            }
            for msg in messages
        ]
    
    def save_preferences(self, chat_id: str, preferences: dict):
        """Save user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO preferences (chat_id, preferences) VALUES (?, ?)',
            (chat_id, json.dumps(preferences))
        )
        conn.commit()
        conn.close()
    
    def get_preferences(self, chat_id: str) -> dict:
        """Get user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT preferences FROM preferences WHERE chat_id = ?',
            (chat_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return {}
    
    def save_chat_title(self, chat_id: str, title: str):
        """Save or update chat title"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO chat_titles (chat_id, title, updated_at) VALUES (?, ?, ?)',
            (chat_id, title, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    
    def get_chat_title(self, chat_id: str) -> Optional[str]:
        """Get chat title"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT title FROM chat_titles WHERE chat_id = ?',
            (chat_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None

class CollegeDataManager:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.colleges = self.load_college_data()
    
    def load_college_data(self) -> List[College]:
        """Load college data from Excel file"""
        try:
            if not os.path.exists(self.excel_path):
                logger.warning(f"Excel file not found at {self.excel_path}")
                return []
            
            df = pd.read_excel(self.excel_path)
            colleges = []
            
            for _, row in df.iterrows():
                college = College(
                    college_id=str(row.get('College ID', '')),
                    name=str(row.get('College', '')),
                    type=str(row.get('Type', '')),
                    affiliation=str(row.get('Affiliation', '')),
                    location=str(row.get('Location', '')),
                    website=str(row.get('Website', '')),
                    contact=str(row.get('Contact', '')),
                    email=str(row.get('E-mail', '')),
                    courses=str(row.get('Courses (ID, Category, Duration, Eligibility, Language, Accreditation, Fees)', '')),
                    scholarship=str(row.get('Scholarship', '')),
                    admission_process=str(row.get('Admission Process', ''))
                )
                colleges.append(college)
            
            logger.info(f"Loaded {len(colleges)} colleges from Excel file")
            return colleges
        except Exception as e:
            logger.error(f"Error loading Excel data: {e}")
            return []
    
    def filter_colleges_by_preferences(self, preferences: UserPreferences) -> List[Dict]:
        """Filter colleges based on user preferences"""
        matching_colleges = []
        
        for college in self.colleges:
            match_score = 0
            match_reasons = []
            
            # Specific Institution Type
            if preferences.specific_institution_type:
                institution_type = preferences.specific_institution_type.upper()
                college_name_upper = college.name.upper()
                
                institution_matches = {
                    'IIT': ['IIT', 'INDIAN INSTITUTE OF TECHNOLOGY'],
                    'NIT': ['NIT', 'NATIONAL INSTITUTE OF TECHNOLOGY'],
                    'IIIT': ['IIIT', 'INDIAN INSTITUTE OF INFORMATION TECHNOLOGY'],
                    'AIIMS': ['AIIMS', 'ALL INDIA INSTITUTE OF MEDICAL SCIENCES'],
                    'IIM': ['IIM', 'INDIAN INSTITUTE OF MANAGEMENT'],
                    'BITS': ['BITS', 'BIRLA INSTITUTE OF TECHNOLOGY'],
                    'VIT': ['VIT', 'VELLORE INSTITUTE OF TECHNOLOGY']
                }
                
                found_match = False
                if institution_type in institution_matches:
                    for pattern in institution_matches[institution_type]:
                        if pattern in college_name_upper:
                            match_score += 50
                            match_reasons.append(f"Matches {institution_type} institution")
                            found_match = True
                            break
                
                if not found_match:
                    continue
            
            # Location filtering
            if preferences.location or preferences.state:
                college_location_lower = college.location.lower()
                location_terms = []
                
                if preferences.location:
                    loc = preferences.location.lower().strip()
                    location_terms.append(loc)
                    
                    region_mappings = {
                        'ncr': ['delhi', 'ncr', 'gurgaon', 'noida', 'ghaziabad'],
                        'bangalore': ['bengaluru', 'bangalore'],
                        'mumbai': ['mumbai', 'bombay']
                    }
                    
                    if loc in region_mappings:
                        location_terms.extend(region_mappings[loc])
                
                if preferences.state:
                    location_terms.append(preferences.state.lower().strip())
                
                location_match = False
                for term in location_terms:
                    if term in college_location_lower:
                        location_match = True
                        match_score += 30
                        match_reasons.append(f"Located in {term}")
                        break
                
                if not location_match and not preferences.specific_institution_type:
                    continue
            
            # Course type filtering
            if preferences.course_type or preferences.specific_course:
                college_courses_lower = college.courses.lower()
                course_terms = []
                
                if preferences.specific_course:
                    course_terms.append(preferences.specific_course.lower())
                if preferences.course_type:
                    course_terms.append(preferences.course_type.lower())
                
                course_match = False
                for term in course_terms:
                    if term in college_courses_lower:
                        course_match = True
                        match_score += 25
                        match_reasons.append(f"Offers {term} courses")
                        break
                
                if not course_match and not preferences.specific_institution_type:
                    continue
            
            # College type filtering
            if preferences.college_type:
                if preferences.college_type.lower() in college.type.lower():
                    match_score += 15
                    match_reasons.append(f"Matches college type: {preferences.college_type}")
            
            if match_score > 0:
                matching_colleges.append({
                    'college': college,
                    'score': match_score,
                    'reasons': match_reasons
                })
        
        matching_colleges.sort(key=lambda x: x['score'], reverse=True)
        return matching_colleges[:10]

class UnifiedAcademicChatbot:
    """Single pipeline chatbot that's conversational and context-aware"""
    
    def __init__(self, openai_api_key: str, excel_path: str, db_path: str, model_name: str = "gpt-4o-mini"):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Single LLM for all operations
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Initialize managers
        self.db_manager = DatabaseManager(db_path)
        self.college_data_manager = CollegeDataManager(excel_path)
        
        # SINGLE UNIFIED MEMORY - maintains context across ALL conversations
        self.chat_memories = defaultdict(lambda: ConversationBufferWindowMemory(
            k=15,
            memory_key="chat_history",
            return_messages=True
        ))
        
        # Setup unified chain and preference extraction
        self._setup_unified_chain()
        self._setup_preference_extraction()
    
    def _setup_unified_chain(self):
        """Setup single unified conversational chain - friend-like, not question-heavy"""
        unified_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Alex, a warm and friendly academic companion. You chat naturally like a supportive friend who genuinely cares.

🎯 YOUR PERSONALITY:
- Talk like a friend, not a formal assistant
- Be warm, encouraging, and relatable
- DON'T bombard with questions - just flow naturally
- Remember everything from the conversation
- Respond directly to what the user asks

💬 CONVERSATION STYLE:
- If someone says "I want to study astrophysics" → Be excited! Share encouragement, maybe mention it's fascinating, and naturally weave in that you can help find colleges if they want
- If they ask for college recommendations → Jump right in with specific suggestions based on what you know
- If they ask follow-up questions about colleges you mentioned → Reference them naturally like "Oh yeah, IIT Delhi that I mentioned earlier..."
- For general questions → Just answer them warmly and directly

🚫 WHAT NOT TO DO:
- DON'T ask "Are you looking for college recommendations or information?" - just respond naturally
- DON'T list multiple options like "I can help you with: 1. 2. 3." unless explicitly asked
- DON'T be overly formal or robotic
- DON'T ask obvious questions - if they say they want to study something, they probably want help with it

✅ WHAT TO DO:
- Be conversational and natural
- Show enthusiasm about their goals
- Offer help smoothly without being pushy
- If college data is in the context, integrate it naturally
- Remember and reference previous parts of the conversation
- Be encouraging and supportive

CONTEXT AWARENESS:
- You maintain full memory of the conversation
- If you recommended colleges earlier, you can discuss them
- If they mentioned preferences before, you remember them
- Be naturally conversational - like texting with a knowledgeable friend

Remember: You're a friend who happens to know a lot about academics and colleges, not a Q&A machine!"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        self.unified_chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.chat_memories[x.get("chat_id", "default")].chat_memory.messages
            )
            | unified_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _setup_preference_extraction(self):
        """Setup preference extraction"""
        self.preference_parser = PydanticOutputParser(pydantic_object=UserPreferences)
        self.preference_prompt = PromptTemplate(
            template="""Extract user preferences for college search from the conversation.

Conversation History:
{conversation_history}

Current Message:
{current_message}

Extract whatever preferences you can find. If nothing specific is mentioned, return null values.

{format_instructions}

Extract preferences as JSON.""",
            input_variables=["conversation_history", "current_message"],
            partial_variables={"format_instructions": self.preference_parser.get_format_instructions()}
        )
        
        self.preference_chain = LLMChain(llm=self.llm, prompt=self.preference_prompt)
    
    def should_get_college_recommendations(self, message: str, chat_history: List) -> bool:
        """Determine if we should fetch college recommendations"""
        message_lower = message.lower().strip()
        
        # Direct recommendation requests
        recommendation_indicators = [
            'recommend college', 'suggest college', 'college recommendation',
            'which college', 'best college', 'top college', 'good college',
            'colleges for', 'colleges in', 'show me college', 'list of college',
            'help me find college', 'looking for college'
        ]
        
        if any(indicator in message_lower for indicator in recommendation_indicators):
            return True
        
        # Check if this is a follow-up about previously recommended colleges
        if chat_history:
            recent_messages = [msg.content for msg in chat_history[-5:] if hasattr(msg, 'content')]
            recent_text = ' '.join(recent_messages).lower()
            
            if 'college' in recent_text or 'university' in recent_text:
                follow_up_indicators = [
                    'tell me more', 'what about', 'first one', 'second one',
                    'that college', 'this college', 'more detail', 'information about'
                ]
                if any(indicator in message_lower for indicator in follow_up_indicators):
                    return True
        
        return False
    
    def extract_preferences(self, chat_id: str, current_message: str) -> UserPreferences:
        """Extract user preferences using LLM"""
        try:
            messages = self.db_manager.get_chat_messages(chat_id)
            conversation_history = "\n".join([
                f"{msg['type'].title()}: {msg['content']}" for msg in messages[-10:]
            ])
            
            result = self.preference_chain.run(
                conversation_history=conversation_history,
                current_message=current_message
            )
            
            try:
                preferences = self.preference_parser.parse(result)
                pref_dict = preferences.dict()
                self.db_manager.save_preferences(chat_id, pref_dict)
                return preferences
            except OutputParserException:
                fixing_parser = OutputFixingParser.from_llm(parser=self.preference_parser, llm=self.llm)
                preferences = fixing_parser.parse(result)
                return preferences
                
        except Exception as e:
            logger.error(f"Error extracting preferences: {e}")
            prev_prefs = self.db_manager.get_preferences(chat_id)
            if prev_prefs:
                return UserPreferences(**prev_prefs)
            return UserPreferences()
    
    def get_openai_recommendations(self, preferences: UserPreferences) -> List[Dict]:
        """Get college recommendations from OpenAI"""
        try:
            pref_parts = []
            
            if preferences.specific_institution_type:
                pref_parts.append(f"Institution type: {preferences.specific_institution_type}")
            if preferences.location:
                pref_parts.append(f"Location: {preferences.location}")
            if preferences.course_type:
                pref_parts.append(f"Course type: {preferences.course_type}")
            if preferences.specific_course:
                pref_parts.append(f"Specific course: {preferences.specific_course}")
            
            if pref_parts:
                preference_text = ", ".join(pref_parts)
                prompt = f"Based on: {preference_text}\n\nRecommend 5 colleges in India."
            else:
                prompt = "Recommend 5 diverse, well-known colleges in India."
            
            prompt += """

Return as JSON array:
[
    {
        "name": "College Name",
        "location": "City, State",
        "type": "Government/Private",
        "courses": "Main courses",
        "features": "Key highlights"
    }
]

Return ONLY the JSON array."""
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            
            result = response.choices[0].message.content.strip()
            
            try:
                colleges = json.loads(result)
                return colleges[:5]
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    colleges = json.loads(json_match.group())
                    return colleges[:5]
                return []
                
        except Exception as e:
            logger.error(f"Error getting OpenAI recommendations: {e}")
            return []
    
    def convert_database_college_to_json(self, college: College, match_score: int, match_reasons: List[str]) -> Dict:
        """Convert database college to JSON format"""
        try:
            # Extract courses
            courses_offered = "Various Programs"
            if college.courses:
                courses_text = college.courses.lower()
                course_list = []
                if 'btech' in courses_text or 'b.tech' in courses_text:
                    course_list.append('B.Tech')
                if 'mtech' in courses_text or 'm.tech' in courses_text:
                    course_list.append('M.Tech')
                if 'mba' in courses_text:
                    course_list.append('MBA')
                if 'mbbs' in courses_text:
                    course_list.append('MBBS')
                if 'bca' in courses_text:
                    course_list.append('BCA')
                if 'mca' in courses_text:
                    course_list.append('MCA')
                if 'data science' in courses_text:
                    course_list.append('Data Science')
                if course_list:
                    courses_offered = ", ".join(course_list)
            
            # Extract fees
            approximate_fees = "Fee information not available"
            if college.courses:
                fee_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d{2})?)', college.courses)
                if fee_match:
                    try:
                        fee_amount = int(fee_match.group(1).replace(',', ''))
                        approximate_fees = f"INR {fee_amount:,} per year"
                    except:
                        approximate_fees = "Fee information not available"
            
            # Notable features
            notable_features_list = []
            if match_reasons:
                notable_features_list.extend(match_reasons[:2])
            if college.scholarship and college.scholarship.lower() != 'nan':
                notable_features_list.append("Scholarship Available")
            if college.type.lower() == 'government':
                notable_features_list.append("Government Institution")
            
            notable_features = ". ".join(notable_features_list[:3]) if notable_features_list else "Quality education institution"
            
            # Format other fields
            admission_process = college.admission_process if college.admission_process and college.admission_process.lower() != 'nan' else "Check official website"
            website = college.website if college.website and college.website.lower() != 'nan' else "Website information not available"
            
            return {
                "id": college.college_id,
                "name": college.name,
                "location": college.location,
                "type": college.type,
                "courses_offered": courses_offered,
                "website": website,
                "admission_process": admission_process,
                "approximate_fees": approximate_fees,
                "notable_features": notable_features,
                "source": "database"
            }
            
        except Exception as e:
            logger.error(f"Error converting database college: {e}")
            return None
    
    def convert_openai_college_to_json(self, college_data: Dict) -> Dict:
        """Convert OpenAI college to JSON format"""
        try:
            return {
                "id": str(uuid.uuid4()),
                "name": college_data.get('name', ''),
                "location": college_data.get('location', ''),
                "type": college_data.get('type', ''),
                "courses_offered": college_data.get('courses', ''),
                "website": "Visit official website for details",
                "admission_process": "Check official website",
                "approximate_fees": "Contact institution for fee details",
                "notable_features": college_data.get('features', ''),
                "source": "openai_knowledge"
            }
            
        except Exception as e:
            logger.error(f"Error converting OpenAI college: {e}")
            return None
    
    def format_college_context(self, filtered_colleges: List[Dict], openai_colleges: List[Dict]) -> str:
        """Format college information as context for the LLM"""
        context_parts = []
        
        # Database colleges
        for i, item in enumerate(filtered_colleges[:5], 1):
            college = item['college']
            context_parts.append(f"""
{i}. {college.name} ({college.location})
   Type: {college.type}
   Courses: {college.courses[:200]}...
   Why it matches: {', '.join(item['reasons'])}
   Website: {college.website}
""")
        
        # OpenAI colleges
        start_idx = len(filtered_colleges) + 1
        for i, college in enumerate(openai_colleges[:5-len(filtered_colleges)], start_idx):
            context_parts.append(f"""
{i}. {college.get('name', 'N/A')} ({college.get('location', 'N/A')})
   Type: {college.get('type', 'N/A')}
   Courses: {college.get('courses', 'N/A')}
   Features: {college.get('features', 'N/A')}
""")
        
        if context_parts:
            return "\n[COLLEGE RECOMMENDATIONS AVAILABLE:\n" + "\n".join(context_parts) + "\n]"
        return ""
    
    def generate_conversation_title(self, message: str, chat_id: str) -> str:
        """Generate conversation title"""
        try:
            messages = self.db_manager.get_chat_messages(chat_id)
            context = ""
            if messages:
                recent_messages = messages[-3:]
                context = " ".join([msg['content'][:100] for msg in recent_messages])
            
            title_prompt = PromptTemplate(
                template="Generate a 3-8 word title for this conversation:\nMessage: {message}\nContext: {context}\nTitle:",
                input_variables=["message", "context"]
            )
            
            title_chain = LLMChain(llm=self.llm, prompt=title_prompt)
            title = title_chain.run(message=message[:200], context=context[:300])
            
            title = title.strip().replace('"', '').replace("'", "")
            if len(title) > 50:
                title = title[:47] + "..."
            
            return title if title else "Academic Discussion"
            
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return "Academic Conversation"
    
    def get_response(self, message: str, chat_id: str) -> Dict[str, Any]:
        """Main unified processing function - conversational and context-aware"""
        timestamp = datetime.now().isoformat()
        
        # Save user message
        self.db_manager.save_message(chat_id, 'human', message, False)
        
        # Generate or retrieve conversation title
        existing_title = self.db_manager.get_chat_title(chat_id)
        conversation_title = existing_title
        
        if not existing_title and len(message.strip()) > 10:
            conversation_title = self.generate_conversation_title(message, chat_id)
            self.db_manager.save_chat_title(chat_id, conversation_title)
        elif not existing_title:
            conversation_title = "New Conversation"
        
        # Get chat history for context
        chat_history = self.chat_memories[chat_id].chat_memory.messages
        
        # Check if we should fetch college recommendations
        should_recommend = self.should_get_college_recommendations(message, chat_history)
        
        # Prepare input for unified chain
        enhanced_message = message
        recommendations_data = []
        
        # If recommendations needed, add college context
        if should_recommend:
            try:
                logger.info("Fetching college recommendations...")
                
                # Extract preferences
                preferences = self.extract_preferences(chat_id, message)
                logger.info(f"Extracted preferences: {preferences}")
                
                # Get college data
                filtered_colleges = self.college_data_manager.filter_colleges_by_preferences(preferences)
                openai_colleges = self.get_openai_recommendations(preferences)
                
                # Convert to JSON format for recommendations array
                for item in filtered_colleges[:5]:
                    json_rec = self.convert_database_college_to_json(
                        item['college'], 
                        item['score'], 
                        item['reasons']
                    )
                    if json_rec:
                        recommendations_data.append(json_rec)
                
                # Add OpenAI colleges to reach 5
                for college in openai_colleges[:5-len(recommendations_data)]:
                    json_rec = self.convert_openai_college_to_json(college)
                    if json_rec:
                        recommendations_data.append(json_rec)
                
                # Add context to message
                college_context = self.format_college_context(filtered_colleges, openai_colleges)
                
                if college_context:
                    enhanced_message = f"{message}\n\n{college_context}"
                    logger.info("College context added to message")
                    
            except Exception as e:
                logger.error(f"Error fetching recommendations: {e}")
        
        # Process through unified chain
        try:
            response = self.unified_chain.invoke({
                "input": enhanced_message,
                "chat_id": chat_id
            })
            
            # Save to unified memory (maintains full context)
            self.chat_memories[chat_id].save_context(
                {"input": message},
                {"output": response}
            )
            
            # Save response to database
            self.db_manager.save_message(chat_id, 'ai', response, should_recommend)
            
            logger.info("✅ Response generated successfully")
            
            return {
                "response": response,
                "is_recommendation": should_recommend,
                "timestamp": timestamp,
                "conversation_title": conversation_title,
                "recommendations": recommendations_data
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I'm having a bit of trouble right now. Could you try asking that again? 😊",
                "is_recommendation": False,
                "timestamp": timestamp,
                "conversation_title": conversation_title,
                "recommendations": []
            }

# Initialize environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "chatbot.db")
EXCEL_PATH = os.getenv("EXCEL_PATH", "colleges.xlsx")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")
    OPENAI_API_KEY = "your-openai-api-key-here"

# Initialize the chatbot
try:
    chatbot = UnifiedAcademicChatbot(OPENAI_API_KEY, EXCEL_PATH, DB_PATH)
    logger.info("✅ Unified Academic Chatbot initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing chatbot: {e}")
    raise

# FastAPI Routes
@app.get("/")
async def root():
    return {
        "message": "Unified Academic Chatbot API - Alex, Your Academic Friend",
        "version": "1.0.0",
        "features": [
            "✅ Friend-like conversational interface",
            "✅ Context-aware conversations",
            "✅ Natural college recommendations",
            "✅ Full conversation memory",
            "✅ No question bombardment"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, chat_id: str = Query(..., description="Chat ID managed by backend")):
    """Unified chat endpoint - conversational and context-aware"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not chat_id.strip():
        raise HTTPException(status_code=400, detail="Chat ID cannot be empty")
    
    try:
        result = chatbot.get_response(
            message=request.message,
            chat_id=chat_id
        )
        return ChatResponse(**result)
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check"""
    try:
        college_count = len(chatbot.college_data_manager.colleges)
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Unified Academic Chatbot API",
            "version": "1.0.0",
            "database": "connected",
            "college_data": f"{college_count} colleges loaded",
            "features": {
                "unified_pipeline": "✅",
                "natural_conversations": "✅",
                "context_awareness": "✅",
                "friend_like_personality": "✅"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    # Validate required environment variables
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
        logger.error("Please set OPENAI_API_KEY environment variable!")
        exit(1)
    
    # Create database if it doesn't exist
    try:
        chatbot.db_manager.init_database()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        exit(1)
    
    # Check if Excel file exists
    if not os.path.exists(EXCEL_PATH):
        logger.warning(f"⚠️ Excel file not found at {EXCEL_PATH}")
        logger.warning("College recommendations will use OpenAI knowledge only")
    else:
        logger.info(f"✅ Excel file loaded: {len(chatbot.college_data_manager.colleges)} colleges")
    
    logger.info("🚀 Starting Unified Academic Chatbot API...")
    logger.info("🎯 Version 1.0.0 - Natural Conversations")
    logger.info("💬 Like chatting with a knowledgeable friend!")
    logger.info("🔗 API: http://localhost:8000")
    logger.info("📚 Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
