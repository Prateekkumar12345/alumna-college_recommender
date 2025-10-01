import os
import logging
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from dataclasses import dataclass, asdict
import re

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import LLMChain, ConversationChain
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
    title="Fixed Dual-Chain Academic Chatbot API",
    description="Academic chatbot with improved intent detection for general conversations and college recommendations",
    version="3.0.0"
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
    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string
        self.init_database()
    
    def get_connection(self):
        """Get a new database connection"""
        return psycopg2.connect(self.db_connection_string)
    
    def init_database(self):
        """Initialize the PostgreSQL database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Create messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    chat_id TEXT,
                    message_type TEXT,
                    content TEXT,
                    is_recommendation BOOLEAN DEFAULT FALSE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS preferences (
                    chat_id TEXT PRIMARY KEY,
                    preferences JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create chat titles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_titles (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("PostgreSQL database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def save_message(self, chat_id: str, message_type: str, content: str, is_recommendation: bool = False):
        """Save a message"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO messages (chat_id, message_type, content, is_recommendation) VALUES (%s, %s, %s, %s)',
                (chat_id, message_type, content, is_recommendation)
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving message: {e}")
    
    def get_chat_messages(self, chat_id: str) -> List[Dict]:
        """Get messages for a chat"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute('''
                SELECT message_type, content, timestamp, is_recommendation
                FROM messages 
                WHERE chat_id = %s 
                ORDER BY timestamp
            ''', (chat_id,))
            messages = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [
                {
                    'type': msg['message_type'],
                    'content': msg['content'],
                    'timestamp': msg['timestamp'].isoformat() if msg['timestamp'] else '',
                    'is_recommendation': msg['is_recommendation']
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []
    
    def save_preferences(self, chat_id: str, preferences: dict):
        """Save user preferences"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO preferences (chat_id, preferences) VALUES (%s, %s) ON CONFLICT (chat_id) DO UPDATE SET preferences = EXCLUDED.preferences',
                (chat_id, json.dumps(preferences))
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
    
    def get_preferences(self, chat_id: str) -> dict:
        """Get user preferences"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'SELECT preferences FROM preferences WHERE chat_id = %s',
                (chat_id,)
            )
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return json.loads(result[0]) if isinstance(result[0], str) else result[0]
            return {}
        except Exception as e:
            logger.error(f"Error getting preferences: {e}")
            return {}
    
    def save_chat_title(self, chat_id: str, title: str):
        """Save or update chat title"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO chat_titles (chat_id, title, updated_at) VALUES (%s, %s, %s) ON CONFLICT (chat_id) DO UPDATE SET title = EXCLUDED.title, updated_at = EXCLUDED.updated_at',
                (chat_id, title, datetime.now())
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving chat title: {e}")
    
    def get_chat_title(self, chat_id: str) -> Optional[str]:
        """Get chat title"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'SELECT title FROM chat_titles WHERE chat_id = %s',
                (chat_id,)
            )
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting chat title: {e}")
            return None

class CollegeDataManager:
    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string
        self.colleges = self.load_college_data()
    
    def get_connection(self):
        """Get a new database connection"""
        return psycopg2.connect(self.db_connection_string)
    
    def load_college_data(self) -> List[College]:
        """Load college data from PostgreSQL database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Query to get all colleges
            cursor.execute('SELECT * FROM college')
            colleges_data = cursor.fetchall()
            cursor.close()
            conn.close()
            
            colleges = []
            for row in colleges_data:
                college = College(
                    college_id=str(row.get('college_id', '')),
                    name=str(row.get('name') or row.get('college_name', '')),
                    type=str(row.get('type', '')),
                    affiliation=str(row.get('affiliation', '')),
                    location=str(row.get('location', '')),
                    website=str(row.get('website', '')),
                    contact=str(row.get('contact', '')),
                    email=str(row.get('email', '')),
                    courses=str(row.get('courses', '')),
                    scholarship=str(row.get('scholarship', '')),
                    admission_process=str(row.get('admission_process', ''))
                )
                colleges.append(college)
            
            logger.info(f"Loaded {len(colleges)} colleges from PostgreSQL database")
            return colleges
        except Exception as e:
            logger.error(f"Error loading college data: {e}")
            return []
    
    def filter_colleges_by_preferences(self, preferences: UserPreferences) -> List[Dict]:
        """Filter colleges based on user preferences with improved matching"""
        matching_colleges = []
        
        for college in self.colleges:
            match_score = 0
            match_reasons = []
            missing_criteria = []
            
            # PRIORITY 1: Specific Institution Type (IIT, NIT, IIIT, etc.)
            if preferences.specific_institution_type:
                institution_type = preferences.specific_institution_type.upper()
                college_name_upper = college.name.upper()
                
                # Check for exact institution type match
                institution_matches = {
                    'IIT': ['IIT', 'INDIAN INSTITUTE OF TECHNOLOGY'],
                    'NIT': ['NIT', 'NATIONAL INSTITUTE OF TECHNOLOGY'],
                    'IIIT': ['IIIT', 'INDIAN INSTITUTE OF INFORMATION TECHNOLOGY'],
                    'AIIMS': ['AIIMS', 'ALL INDIA INSTITUTE OF MEDICAL SCIENCES'],
                    'IIM': ['IIM', 'INDIAN INSTITUTE OF MANAGEMENT'],
                    'BITS': ['BITS', 'BIRLA INSTITUTE OF TECHNOLOGY'],
                    'THAPAR': ['THAPAR'],
                    'VIT': ['VIT', 'VELLORE INSTITUTE OF TECHNOLOGY'],
                    'SRM': ['SRM'],
                    'MANIPAL': ['MANIPAL']
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
                    missing_criteria.append(f"Not a {institution_type} institution")
                    continue
            
            # PRIORITY 2: Location filtering
            location_match = True
            if preferences.location:
                location_terms = [preferences.location.lower()]
                if preferences.state:
                    location_terms.append(preferences.state.lower())
                
                college_location = college.location.lower()
                location_match = False
                for term in location_terms:
                    if term in college_location:
                        location_match = True
                        match_score += 25
                        match_reasons.append(f"Located in {preferences.location}")
                        break
                
                if not location_match:
                    missing_criteria.append(f"Not in preferred location: {preferences.location}")
                    if not preferences.specific_institution_type:
                        continue
            
            # PRIORITY 3: College type filtering
            if preferences.college_type:
                if preferences.college_type.lower() in college.type.lower():
                    match_score += 20
                    match_reasons.append(f"Matches college type: {preferences.college_type}")
                else:
                    missing_criteria.append(f"Not a {preferences.college_type} college")
                    if not preferences.specific_institution_type:
                        continue
            
            # PRIORITY 4: Course type filtering
            course_match = True
            if preferences.course_type or preferences.specific_course:
                college_courses = college.courses.lower()
                
                if preferences.specific_course:
                    course_terms = [preferences.specific_course.lower()]
                    if preferences.course_type:
                        course_terms.append(preferences.course_type.lower())
                else:
                    course_terms = [preferences.course_type.lower()]
                
                course_match = False
                for term in course_terms:
                    if term in college_courses:
                        course_match = True
                        match_score += 20
                        match_reasons.append(f"Offers {term} courses")
                        break
                
                if not course_match:
                    missing_criteria.append(f"Doesn't offer preferred course type")
                    if not preferences.specific_institution_type:
                        continue
            
            # PRIORITY 5: Level filtering
            if preferences.level:
                if preferences.level.lower() in college.courses.lower():
                    match_score += 10
                    match_reasons.append(f"Offers {preferences.level} programs")
            
            if match_score > 0 or preferences.specific_institution_type:
                matching_colleges.append({
                    'college': college,
                    'score': match_score,
                    'reasons': match_reasons,
                    'missing': missing_criteria
                })
        
        matching_colleges.sort(key=lambda x: x['score'], reverse=True)
        return matching_colleges[:10]

class SmartIntentDetector:
    """Enhanced intent detection with precise classification"""
    
    def __init__(self, llm):
        self.llm = llm
        self._setup_intent_classifier()
    
    def _setup_intent_classifier(self):
        """Setup the intent classification chain"""
        intent_prompt = PromptTemplate(
            template="""Analyze the following user message and classify its intent precisely.

USER MESSAGE: "{message}"

INTENT CATEGORIES:
1. GREETING - Simple greetings, casual conversation starters (hi, hello, how are you)
2. ACADEMIC_HELP - Questions about studies, subjects, exams, learning strategies, homework help
3. COLLEGE_RECOMMENDATION - Explicitly asking for college/university recommendations or suggestions
4. CAREER_GUIDANCE - General career advice, "what should I do after 12th", pathway discussions
5. DUAL_INTENT - Explicitly requesting both guidance AND specific college recommendations

RULES FOR CLASSIFICATION:
- COLLEGE_RECOMMENDATION: Must contain explicit request words like "recommend", "suggest", "which college", "list colleges", "show me colleges"
- ACADEMIC_HELP: Questions about studying, subjects, concepts, exam preparation
- CAREER_GUIDANCE: General advice about career paths, options after education
- GREETING: Short social interactions, introductions
- DUAL_INTENT: Only when BOTH career guidance AND college recommendations are explicitly requested

Respond with ONLY the category name: GREETING, ACADEMIC_HELP, COLLEGE_RECOMMENDATION, CAREER_GUIDANCE, or DUAL_INTENT""",
            input_variables=["message"]
        )
        
        self.intent_chain = LLMChain(llm=self.llm, prompt=intent_prompt)
    
    def detect_intent(self, message: str) -> Dict[str, bool]:
        """Detect user intent with high precision"""
        try:
            intent_result = self.intent_chain.run(message=message).strip().upper()
            logger.info(f"LLM Intent Classification: {intent_result} for message: '{message}'")
            
            if intent_result == "GREETING":
                return {'academic': True, 'recommendation': False}
            elif intent_result == "ACADEMIC_HELP":
                return {'academic': True, 'recommendation': False}
            elif intent_result == "COLLEGE_RECOMMENDATION":
                return {'academic': False, 'recommendation': True}
            elif intent_result == "CAREER_GUIDANCE":
                return {'academic': True, 'recommendation': False}
            elif intent_result == "DUAL_INTENT":
                return {'academic': True, 'recommendation': True}
            else:
                return self._fallback_detection(message)
                
        except Exception as e:
            logger.error(f"Intent detection error: {e}")
            return self._fallback_detection(message)
    
    def _fallback_detection(self, message: str) -> Dict[str, bool]:
        """Fallback rule-based intent detection"""
        message_lower = message.lower().strip()
        
        strong_rec_indicators = [
            'recommend college', 'suggest college', 'recommend university', 'suggest university',
            'which college should', 'which university should', 'best college for', 'best university for',
            'list of college', 'list of university', 'colleges in', 'universities in',
            'show me college', 'show me university', 'college recommendation', 'university recommendation',
            'help me find college', 'help me find university', 'looking for college', 'looking for university'
        ]
        
        if any(indicator in message_lower for indicator in strong_rec_indicators):
            return {'academic': False, 'recommendation': True}
        
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(message_lower.startswith(greeting) for greeting in greetings):
            return {'academic': True, 'recommendation': False}
        
        return {'academic': True, 'recommendation': False}

class DualChainAcademicChatbot:
    def __init__(self, openai_api_key: str, db_connection_string: str, model_name: str = "gpt-3.5-turbo"):
        """Initialize the dual-chain chatbot with improved intent detection"""
        
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Initialize LLMs
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            max_tokens=500
        )
        
        self.academic_llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.3,
            max_tokens=500
        )
        
        # Initialize managers
        self.db_manager = DatabaseManager(db_connection_string)
        self.college_data_manager = CollegeDataManager(db_connection_string)
        self.intent_detector = SmartIntentDetector(self.llm)
        
        # Memory for both chains
        self.chat_memories = defaultdict(lambda: ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True
        ))
        
        self.recommendation_memories = defaultdict(lambda: ConversationBufferWindowMemory(
            k=5,
            memory_key="recommendation_history",
            return_messages=True
        ))
        
        # Setup chains
        self._setup_academic_chain()
        self._setup_recommendation_chain()
        self._setup_fusion_chain()
        self._setup_preference_extraction()
    
    def _setup_academic_chain(self):
        """Setup academic conversation chain"""
        academic_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Alex, a friendly and knowledgeable academic assistant. You help with:
- General academic questions and explanations across all subjects
- Study strategies and learning techniques
- Career guidance and educational pathways (general advice only, not specific college recommendations)
- Subject-specific help (math, science, literature, etc.)
- Exam preparation advice and study planning
- Research methodologies and academic writing
- Educational concept explanations
- Friendly greetings and casual conversations about learning

Important Guidelines:
- For greetings: Respond warmly and ask how you can help with their learning journey
- For career guidance: Provide general advice about paths, skills, and opportunities
- For academic help: Give detailed explanations with examples and actionable tips
- DO NOT provide specific college recommendations - that's handled by a specialist

Personality: Warm, encouraging, patient, conversational, and focused on educational growth."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        self.academic_chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.chat_memories[x.get("chat_id", "default")].chat_memory.messages
            )
            | academic_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _setup_recommendation_chain(self):
        """Setup college recommendation chain"""
        recommendation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized college admission counselor focused ONLY on providing specific college and university recommendations. Your expertise includes:
- Specific college and university recommendations across India
- Detailed information about admission processes and entrance exams
- Course details, fee structures, and program specifics
- College comparisons and institutional rankings
- Scholarship and financial aid opportunities
- Campus facilities and placement records

Important Guidelines:
- Provide concrete, actionable college recommendations with specific names
- Include practical details like fees, admission processes, and course offerings
- Focus on matching colleges to user preferences and requirements
- Be factual and specific rather than general
- Always aim to recommend actual institutions when possible

DO NOT provide general career advice - focus exclusively on college recommendations."""),
            MessagesPlaceholder(variable_name="recommendation_history"),
            ("human", "{input}"),
        ])
        
        self.recommendation_chain = (
            RunnablePassthrough.assign(
                recommendation_history=lambda x: self.recommendation_memories[x.get("chat_id", "default")].chat_memory.messages
            )
            | recommendation_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _setup_fusion_chain(self):
        """Setup response fusion chain for dual intent queries"""
        self.fusion_prompt = PromptTemplate(
            template="""
            COMBINE RESPONSES FROM TWO SPECIALISTS FOR A DUAL-INTENT QUERY:
            
            USER QUERY: {user_input}
            
            ACADEMIC GUIDANCE (General Education & Career Advice):
            {academic_response}
            
            COLLEGE SPECIALIST (Specific Recommendations):
            {recommendation_response}
            
            TASK:
            Create a comprehensive response that:
            1. Starts with the general academic/career guidance
            2. Smoothly transitions to specific college recommendations
            3. Maintains natural flow and removes redundancy
            4. Provides both conceptual guidance AND actionable college options
            5. Uses clear sections or transitions between the two aspects
            
            COMBINED RESPONSE:
            """,
            input_variables=["academic_response", "recommendation_response", "user_input"]
        )
        
        self.fusion_chain = (
            {
                "academic_response": RunnablePassthrough(),
                "recommendation_response": RunnablePassthrough(),
                "user_input": RunnablePassthrough()
            }
            | self.fusion_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _setup_preference_extraction(self):
        """Setup preference extraction"""
        self.preference_parser = PydanticOutputParser(pydantic_object=UserPreferences)
        self.preference_prompt = PromptTemplate(
            template="""
            Extract user preferences for college search from the conversation. Pay special attention to specific institution types.
            
            Conversation History:
            {conversation_history}
            
            Current Message:
            {current_message}
            
            IMPORTANT: Look for specific institution types like:
            - IIT (Indian Institute of Technology)
            - NIT (National Institute of Technology) 
            - IIIT (Indian Institute of Information Technology)
            - AIIMS (All India Institute of Medical Sciences)
            - IIM (Indian Institute of Management)
            - BITS, VIT, SRM, Manipal, etc.
            
            If the user mentions any of these, set the specific_institution_type field accordingly.
            
            {format_instructions}
            
            Extract preferences as JSON. Use null for fields without clear preferences.
            """,
            input_variables=["conversation_history", "current_message"],
            partial_variables={"format_instructions": self.preference_parser.get_format_instructions()}
        )
        
        self.preference_chain = LLMChain(llm=self.academic_llm, prompt=self.preference_prompt)
    
    def extract_preferences_with_llm(self, chat_id: str, current_message: str) -> UserPreferences:
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
            except OutputParserException as e:
                logger.error(f"Parser error: {e}")
                fixing_parser = OutputFixingParser.from_llm(parser=self.preference_parser, llm=self.llm)
                preferences = fixing_parser.parse(result)
                return preferences
                
        except Exception as e:
            logger.error(f"Error extracting preferences: {e}")
            prev_prefs = self.db_manager.get_preferences(chat_id)
            if prev_prefs:
                return UserPreferences(**prev_prefs)
            return UserPreferences()
    
    def get_openai_college_recommendations(self, preferences: UserPreferences, location: str = None) -> List[Dict]:
        """Get college recommendations from OpenAI with improved handling of specific institutions"""
        try:
            pref_parts = []
            
            if preferences.specific_institution_type:
                institution_type = preferences.specific_institution_type.upper()
                if institution_type == 'IIT':
                    pref_parts.append("IIT (Indian Institute of Technology) colleges only")
                elif institution_type == 'NIT':
                    pref_parts.append("NIT (National Institute of Technology) colleges only")
                elif institution_type == 'IIIT':
                    pref_parts.append("IIIT (Indian Institute of Information Technology) colleges only")
                else:
                    pref_parts.append(f"{institution_type} institutions only")
            
            if location:
                pref_parts.append(f"Location: {location}")
            if preferences.course_type:
                pref_parts.append(f"Course type: {preferences.course_type}")
            if preferences.specific_course:
                pref_parts.append(f"Specific course: {preferences.specific_course}")
            if preferences.college_type:
                pref_parts.append(f"College type: {preferences.college_type}")
            if preferences.level:
                pref_parts.append(f"Level: {preferences.level}")
            
            preference_text = ", ".join(pref_parts) if pref_parts else "General preferences"
            
            if preferences.specific_institution_type:
                institution_type = preferences.specific_institution_type.upper()
                if institution_type == 'IIT':
                    prompt = f"""
                    List 5-7 IIT (Indian Institute of Technology) colleges in India for B.Tech programs.
                    
                    Provide response as a JSON array with this exact structure:
                    [
                        {{
                            "id": "iit_college_id",
                            "name": "IIT College Name",
                            "location": "City, State",
                            "type": "Government",
                            "courses_offered": "B.Tech specializations offered",
                            "website": "Official IIT website",
                            "admission_process": "JEE Advanced + Counselling",
                            "approximate_fees": "Fee range per year",
                            "notable_features": "Key highlights of this IIT",
                            "source": "openai_knowledge"
                        }}
                    ]
                    
                    Only include actual IIT colleges. Return only JSON:
                    """
                elif institution_type == 'NIT':
                    prompt = f"""
                    List 5-7 NIT (National Institute of Technology) colleges in India for engineering programs.
                    
                    Provide response as a JSON array with the exact structure shown above.
                    Only include actual NIT colleges. Return only JSON:
                    """
                else:
                    prompt = f"""
                    Recommend 3-5 {institution_type} colleges/universities in India based on: {preference_text}
                    
                    Provide response as a JSON array with this structure:
                    [
                        {{
                            "id": "unique_id",
                            "name": "College Name",
                            "location": "City, State",
                            "type": "Government/Private/Deemed",
                            "courses_offered": "Main courses",
                            "website": "Official website",
                            "admission_process": "Admission description",
                            "approximate_fees": "Fee range",
                            "notable_features": "Key highlights",
                            "source": "openai_knowledge"
                        }}
                    ]
                    
                    Return only JSON:
                    """
            else:
                prompt = f"""
                Recommend 3-5 good colleges/universities in India based on: {preference_text}
                
                Provide response as a JSON array with this structure:
                [
                    {{
                        "id": "unique_id",
                        "name": "College Name",
                        "location": "City, State",
                        "type": "Government/Private/Deemed",
                        "courses_offered": "Main courses",
                        "website": "Official website",
                        "admission_process": "Admission description",
                        "approximate_fees": "Fee range",
                        "notable_features": "Key highlights",
                        "source": "openai_knowledge"
                    }}
                ]
                
                Return only JSON:
                """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return []
                
        except Exception as e:
            logger.error(f"Error getting OpenAI recommendations: {e}")
            return []
    
    def convert_database_college_to_json(self, college: College, match_score: int, match_reasons: List[str]) -> Dict:
        """Convert database college to JSON format"""
        try:
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
                if course_list:
                    courses_offered = ", ".join(course_list)
            
            approximate_fees = "Fee information not available"
            if college.courses:
                fee_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d{2})?)', college.courses)
                if fee_match:
                    try:
                        fee_amount = int(fee_match.group(1).replace(',', ''))
                        approximate_fees = f"INR {fee_amount:,} per year"
                    except:
                        approximate_fees = "Fee information not available"
            
            notable_features_list = []
            if match_reasons:
                notable_features_list.extend(match_reasons[:2])
            if college.scholarship and college.scholarship.lower() != 'nan':
                notable_features_list.append("Scholarship Available")
            if college.type.lower() == 'government':
                notable_features_list.append("Government Institution")
            
            notable_features = ". ".join(notable_features_list[:3]) if notable_features_list else "Quality education institution"
            
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
                "id": college_data.get('id', str(uuid.uuid4())),
                "name": college_data.get('name', ''),
                "location": college_data.get('location', ''),
                "type": college_data.get('type', ''),
                "courses_offered": college_data.get('courses_offered', ''),
                "website": college_data.get('website', ''),
                "admission_process": college_data.get('admission_process', ''),
                "approximate_fees": college_data.get('approximate_fees', ''),
                "notable_features": college_data.get('notable_features', ''),
                "source": college_data.get('source', 'openai_knowledge')
            }
            
        except Exception as e:
            logger.error(f"Error converting OpenAI college: {e}")
            return None
    
    def format_college_recommendations(self, filtered_colleges: List[Dict], openai_colleges: List[Dict], preferences: UserPreferences) -> Tuple[List[Dict], str]:
        """Format college recommendations with clear prioritization of database results"""
        recommendations = []
        database_count = 0
        openai_count = 0
        
        logger.info(f"Processing {len(filtered_colleges)} database colleges...")
        for item in filtered_colleges:
            college = item['college']
            json_rec = self.convert_database_college_to_json(college, item['score'], item['reasons'])
            if json_rec:
                recommendations.append(json_rec)
                database_count += 1
                logger.info(f"Added database college: {json_rec['name']}")
        
        if len(recommendations) < 5 and openai_colleges:
            needed = min(5 - len(recommendations), len(openai_colleges))
            logger.info(f"Adding {needed} OpenAI colleges to supplement database results...")
            for college in openai_colleges[:needed]:
                json_rec = self.convert_openai_college_to_json(college)
                if json_rec:
                    recommendations.append(json_rec)
                    openai_count += 1
                    logger.info(f"Added OpenAI college: {json_rec['name']}")
        
        if recommendations:
            source_info = []
            if database_count > 0:
                source_info.append(f"{database_count} from database")
            if openai_count > 0:
                source_info.append(f"{openai_count} from AI knowledge")
            
            source_text = " (" + ", ".join(source_info) + ")" if source_info else ""
            
            text_summary = f"Found {len(recommendations)} colleges matching your preferences{source_text}:"
            
            for i, rec in enumerate(recommendations, 1):
                source_indicator = " 🗄️" if rec['source'] == 'database' else " 🤖"
                text_summary += f"\n{i}. {rec['name']} - {rec['location']} ({rec['type']}){source_indicator}"
        else:
            if preferences.specific_institution_type:
                text_summary = f"No {preferences.specific_institution_type} colleges found in database matching your criteria. You may need to adjust your preferences or the database may not contain these specific institutions."
            else:
                text_summary = "No colleges found matching your current preferences. Try adjusting your criteria."
        
        logger.info(f"Final recommendations: {len(recommendations)} total ({database_count} database, {openai_count} OpenAI)")
        return recommendations, text_summary
    
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
        """Main processing function with improved intent detection"""
        timestamp = datetime.now().isoformat()
        
        self.db_manager.save_message(chat_id, 'human', message, False)
        
        existing_title = self.db_manager.get_chat_title(chat_id)
        conversation_title = existing_title
        
        if not existing_title and len(message.strip()) > 10:
            conversation_title = self.generate_conversation_title(message, chat_id)
            self.db_manager.save_chat_title(chat_id, conversation_title)
        elif not existing_title:
            conversation_title = "New Conversation"
        
        intent = self.intent_detector.detect_intent(message)
        logger.info(f"Query: '{message}' | Detected intent: {intent}")
        
        academic_response = ""
        recommendation_response = ""
        recommendations_data = []
        is_recommendation = False
        
        if intent['academic']:
            try:
                logger.info("Processing through Academic Chain...")
                academic_response = self.academic_chain.invoke({
                    "input": message,
                    "chat_id": chat_id
                })
                self.chat_memories[chat_id].save_context(
                    {"input": message},
                    {"output": academic_response}
                )
                logger.info("✅ Academic chain completed successfully")
            except Exception as e:
                logger.error(f"❌ Academic chain error: {e}")
                academic_response = "I'm having trouble processing your request. Please try again."
        
        if intent['recommendation']:
            try:
                logger.info("Processing through Recommendation Chain...")
                
                preferences = self.extract_preferences_with_llm(chat_id, message)
                logger.info(f"Extracted preferences: {preferences}")
                
                filtered_colleges = self.college_data_manager.filter_colleges_by_preferences(preferences)
                logger.info(f"Database colleges found: {len(filtered_colleges)}")
                
                openai_colleges = []
                if len(filtered_colleges) < 3:
                    logger.info("Insufficient database results, getting OpenAI recommendations...")
                    openai_colleges = self.get_openai_college_recommendations(preferences, preferences.location)
                    logger.info(f"OpenAI colleges found: {len(openai_colleges)}")
                else:
                    logger.info("Sufficient database results found, skipping OpenAI recommendations")
                
                recommendations_data, summary_text = self.format_college_recommendations(
                    filtered_colleges, openai_colleges, preferences
                )
                
                is_recommendation = len(recommendations_data) > 0
                logger.info(f"Total recommendations prepared: {len(recommendations_data)}")
                
                recommendation_input = f"User is asking for college recommendations: {message}"
                if recommendations_data:
                    recommendation_input += f"\n\nI have found {len(recommendations_data)} matching colleges. Please provide a helpful response that guides the user about these college options and mentions that detailed information is available in the recommendations section."
                
                recommendation_response = self.recommendation_chain.invoke({
                    "input": recommendation_input,
                    "chat_id": chat_id
                })
                
                self.recommendation_memories[chat_id].save_context(
                    {"input": message},
                    {"output": recommendation_response}
                )
                
                logger.info("✅ Recommendation chain completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Recommendation chain error: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                recommendation_response = "I encountered an issue while finding college recommendations. Please try rephrasing your request."
        
        final_response = self._create_final_response(
            academic_response, recommendation_response, message, intent
        )
        
        self.db_manager.save_message(chat_id, 'ai', final_response, is_recommendation)
        
        return {
            "response": final_response,
            "is_recommendation": is_recommendation,
            "timestamp": timestamp,
            "conversation_title": conversation_title,
            "recommendations": recommendations_data
        }
    
    def _create_final_response(self, academic_response: str, recommendation_response: str, 
                              user_input: str, intent: Dict[str, bool]) -> str:
        """Create final response based on intent and available responses"""
        
        if not academic_response and not recommendation_response:
            return "I apologize, but I encountered an error while processing your request. Please try again, and I'll do my best to help!"
        
        if intent['academic'] and not intent['recommendation']:
            return academic_response or "I'm here to help with your academic questions. Please try asking again."
        
        if intent['recommendation'] and not intent['academic']:
            return recommendation_response or "I can help you find colleges. Please provide more details about your preferences."
        
        if intent['academic'] and intent['recommendation']:
            if academic_response and recommendation_response:
                try:
                    logger.info("Fusing responses for dual intent query...")
                    fused_response = self.fusion_chain.invoke({
                        "academic_response": academic_response,
                        "recommendation_response": recommendation_response,
                        "user_input": user_input
                    })
                    return fused_response
                except Exception as e:
                    logger.error(f"Fusion error: {e}")
                    return f"{academic_response}\n\n**College Recommendations:**\n{recommendation_response}"
            elif academic_response:
                return academic_response
            elif recommendation_response:
                return recommendation_response
        
        return academic_response or recommendation_response or "I'm here to help! Please ask me about academics or college recommendations."

# Initialize environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables")
    exit(1)

if not DB_CONNECTION_STRING:
    logger.error("DB_CONNECTION_STRING not found in environment variables")
    exit(1)

# Initialize the dual-chain chatbot
try:
    chatbot = DualChainAcademicChatbot(OPENAI_API_KEY, DB_CONNECTION_STRING)
    logger.info("✅ Fixed Dual-Chain Academic Chatbot with PostgreSQL initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing chatbot: {e}")
    raise

# FastAPI Routes
@app.get("/")
async def root():
    return {
        "message": "Fixed Dual-Chain Academic Chatbot API with PostgreSQL!",
        "version": "3.0.0",
        "features": [
            "🎯 Smart Intent Detection with LLM Classification",
            "🎓 Academic Conversations & Guidance", 
            "🏫 Precise College Recommendations",
            "🧠 Context-Aware Response Generation",
            "🔀 Intelligent Response Fusion",
            "💬 Improved Greeting Handling",
            "🐘 PostgreSQL Database Integration"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, chat_id: str = Query(..., description="Chat ID managed by backend")):
    """Enhanced chat endpoint with precise intent detection"""
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
    """Detailed health check"""
    try:
        college_count = len(chatbot.college_data_manager.colleges)
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Fixed Dual-Chain Academic Chatbot API with PostgreSQL",
            "version": "3.0.0",
            "database": "PostgreSQL connected",
            "college_data": f"{college_count} colleges loaded",
            "features": {
                "smart_intent_detection": "✅ LLM + Rule-based",
                "academic_chain": "✅ Active",
                "recommendation_chain": "✅ Active",
                "response_fusion": "✅ Active",
                "context_awareness": "✅ Active",
                "greeting_handling": "✅ Enhanced",
                "postgresql_integration": "✅ Active"
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
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
        logger.error("Please set OPENAI_API_KEY environment variable!")
        exit(1)
    
    if not DB_CONNECTION_STRING:
        logger.error("Please set DB_CONNECTION_STRING environment variable!")
        exit(1)
    
    # Test database connection
    try:
        test_conn = psycopg2.connect(DB_CONNECTION_STRING)
        test_conn.close()
        logger.info("✅ PostgreSQL connection successful")
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        exit(1)
    
    # Create database tables
    try:
        chatbot.db_manager.init_database()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        exit(1)
    
    # Check college data
    college_count = len(chatbot.college_data_manager.colleges)
    if college_count == 0:
        logger.warning("⚠️ No colleges found in database")
        logger.warning("College recommendations will use OpenAI knowledge only")
    else:
        logger.info(f"✅ Loaded {college_count} colleges from PostgreSQL database")
    
    logger.info("🚀 Starting Fixed Dual-Chain Academic Chatbot API with PostgreSQL...")
    logger.info("🎯 Features: Smart Intent Detection + Academic Help + College Recommendations + PostgreSQL")
    logger.info("🔗 API: http://localhost:8000")
    logger.info("📚 Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )