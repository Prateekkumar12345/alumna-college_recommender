
import os
import logging
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import uuid
import sqlite3
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
    title="Dynamic Academic Chatbot API",
    description="Academic chatbot that provides college recommendations based on available information",
    version="4.1.0"
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
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create messages table
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
        
        # Create preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preferences (
                chat_id TEXT PRIMARY KEY,
                preferences TEXT,
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
        """Filter colleges based on user preferences with FIXED location matching"""
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
                            match_score += 50  # Highest priority
                            match_reasons.append(f"Matches {institution_type} institution")
                            found_match = True
                            break
                
                # If specific institution type requested but not found, skip this college
                if not found_match:
                    missing_criteria.append(f"Not a {institution_type} institution")
                    continue
            
            # PRIORITY 2: FIXED Location filtering with flexible matching
            location_match = True
            if preferences.location or preferences.state:
                college_location_lower = college.location.lower()
                location_terms = []
                
                # Build comprehensive location search terms
                if preferences.location:
                    loc = preferences.location.lower().strip()
                    location_terms.append(loc)
                    
                    # Handle common regional terms
                    region_mappings = {
                        'national capital region': ['delhi', 'ncr', 'gurgaon', 'noida', 'ghaziabad', 'faridabad'],
                        'ncr': ['delhi', 'ncr', 'gurgaon', 'noida', 'ghaziabad', 'faridabad'],
                        'bangalore': ['bengaluru', 'bangalore'],
                        'bengaluru': ['bengaluru', 'bangalore'],
                        'bombay': ['mumbai', 'bombay'],
                        'mumbai': ['mumbai', 'bombay'],
                        'madras': ['chennai', 'madras'],
                        'chennai': ['chennai', 'madras']
                    }
                    
                    if loc in region_mappings:
                        location_terms.extend(region_mappings[loc])
                
                if preferences.state:
                    state = preferences.state.lower().strip()
                    location_terms.append(state)
                
                # Remove duplicates
                location_terms = list(set(location_terms))
                
                location_match = False
                matched_term = None
                for term in location_terms:
                    if term in college_location_lower:
                        location_match = True
                        matched_term = term
                        match_score += 30  # Increased priority
                        match_reasons.append(f"Located in {matched_term}")
                        break
                
                if not location_match:
                    missing_criteria.append(f"Not in preferred location")
                    # Skip if location was specified but not matched
                    if not preferences.specific_institution_type:
                        continue
            
            # PRIORITY 3: Course type filtering with IMPROVED matching
            course_match = True
            if preferences.course_type or preferences.specific_course:
                college_courses_lower = college.courses.lower()
                
                # Build comprehensive course search terms
                course_terms = []
                if preferences.specific_course:
                    course_terms.append(preferences.specific_course.lower())
                if preferences.course_type:
                    course_terms.append(preferences.course_type.lower())
                
                # Add variations for common courses
                course_variations = {
                    'data science': ['data science', 'data analytics', 'artificial intelligence', 'machine learning', 'ai', 'ml'],
                    'engineering': ['engineering', 'btech', 'b.tech', 'be', 'b.e.'],
                    'management': ['management', 'mba', 'm.b.a', 'pgdm'],
                    'medicine': ['medicine', 'mbbs', 'md', 'medical'],
                    'computer science': ['computer science', 'cs', 'cse', 'computer engineering']
                }
                
                expanded_terms = []
                for term in course_terms:
                    expanded_terms.append(term)
                    if term in course_variations:
                        expanded_terms.extend(course_variations[term])
                
                course_match = False
                matched_course = None
                for term in expanded_terms:
                    if term in college_courses_lower:
                        course_match = True
                        matched_course = term
                        match_score += 25
                        match_reasons.append(f"Offers {term} courses")
                        break
                
                if not course_match:
                    missing_criteria.append(f"Doesn't offer required course")
                    # Skip if course was specified but not matched
                    if not preferences.specific_institution_type:
                        continue
            
            # PRIORITY 4: College type filtering
            if preferences.college_type:
                college_type_match = preferences.college_type.lower() in college.type.lower()
                if college_type_match:
                    match_score += 15
                    match_reasons.append(f"Matches college type: {preferences.college_type}")
                else:
                    missing_criteria.append(f"Not a {preferences.college_type} college")
                    if not preferences.specific_institution_type:
                        continue
            
            # PRIORITY 5: Level filtering
            if preferences.level:
                level_terms = {
                    'ug': ['ug', 'undergraduate', 'bachelor', 'btech', 'be', 'bsc', 'ba', 'bcom'],
                    'pg': ['pg', 'postgraduate', 'master', 'mtech', 'me', 'msc', 'ma', 'mcom', 'mba']
                }
                
                level_lower = preferences.level.lower()
                if level_lower in level_terms:
                    if any(term in college.courses.lower() for term in level_terms[level_lower]):
                        match_score += 10
                        match_reasons.append(f"Offers {preferences.level} programs")
            
            # Only include colleges with reasonable match or if specific institution type was requested
            if match_score > 0:
                matching_colleges.append({
                    'college': college,
                    'score': match_score,
                    'reasons': match_reasons,
                    'missing': missing_criteria
                })
        
        # Sort by match score (highest first)
        matching_colleges.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top 15 to ensure we have enough for combination with OpenAI results
        return matching_colleges[:15]

class SmartIntentDetector:
    """Dynamic intent detection that allows recommendations with minimal info"""
    
    def __init__(self, llm):
        self.llm = llm
        self._setup_intent_classifier()
    
    def _setup_intent_classifier(self):
        """Setup the intent classification chain"""
        intent_prompt = PromptTemplate(
            template="""Analyze the following user message and classify its intent.

USER MESSAGE: "{message}"

INTENT CATEGORIES:
1. GREETING - Simple greetings only (hi, hello, how are you)
2. GENERAL_CHAT - General conversation NOT related to education/college (relationships, life advice, casual topics, technical questions about the chatbot itself like "which LLM do you use")
3. ACADEMIC_HELP - Questions about studying, subjects, exams, learning strategies, homework, educational courses
4. CAREER_GUIDANCE - Career advice, "what should I do after 12th", pathway discussions
5. COLLEGE_RECOMMENDATION - When asking for college/university recommendations, even with minimal criteria
6. GENERAL_COLLEGE_INFO - General questions about colleges, education system, etc.

IMPORTANT RULES:
- GENERAL_CHAT: Questions about relationships, personal life, which LLM/technology chatbot uses, how the bot works
- ACADEMIC_HELP: Questions about courses to study, learning paths, subjects, exam preparation
- COLLEGE_RECOMMENDATION: Any request for college suggestions, even vague ones like "recommend colleges" - we'll provide general recommendations
- CAREER_GUIDANCE: General career advice
- GENERAL_COLLEGE_INFO: Questions about college life, admission processes, etc.

EXAMPLES:
- "How to propose a girl" → GENERAL_CHAT
- "Which LLM do you work on" → GENERAL_CHAT  
- "Recommend me courses to study abroad" → ACADEMIC_HELP
- "Please recommend me colleges" → COLLEGE_RECOMMENDATION
- "Suggest colleges for engineering in Delhi" → COLLEGE_RECOMMENDATION
- "What should I study to become a data scientist" → ACADEMIC_HELP
- "Tell me about college life" → GENERAL_COLLEGE_INFO

Respond with ONLY ONE category name.""",
            input_variables=["message"]
        )
        
        self.intent_chain = LLMChain(llm=self.llm, prompt=intent_prompt)
    
    def detect_intent(self, message: str) -> Dict[str, Any]:
        """Detect user intent with dynamic classification"""
        try:
            # Get LLM classification
            intent_result = self.intent_chain.run(message=message).strip().upper()
            logger.info(f"LLM Intent Classification: {intent_result} for message: '{message}'")
            
            # Map to boolean flags - REMOVED "needs_more_info" logic
            if intent_result == "GREETING":
                return {'academic': True, 'recommendation': False}
            elif intent_result == "GENERAL_CHAT":
                return {'academic': True, 'recommendation': False}
            elif intent_result == "ACADEMIC_HELP":
                return {'academic': True, 'recommendation': False}
            elif intent_result == "CAREER_GUIDANCE":
                return {'academic': True, 'recommendation': False}
            elif intent_result == "COLLEGE_RECOMMENDATION":
                return {'academic': False, 'recommendation': True}
            elif intent_result == "GENERAL_COLLEGE_INFO":
                return {'academic': True, 'recommendation': False}
            else:
                # Fallback to rule-based detection
                return self._fallback_detection(message)
                
        except Exception as e:
            logger.error(f"Intent detection error: {e}")
            return self._fallback_detection(message)
    
    def _fallback_detection(self, message: str) -> Dict[str, Any]:
        """Dynamic fallback rule-based intent detection - NO preference asking"""
        message_lower = message.lower().strip()
        
        # RULE 1: Check for general chat indicators (highest priority)
        general_chat_indicators = [
            'how to propose', 'relationship', 'girlfriend', 'boyfriend', 
            'love', 'dating', 'which llm', 'which model', 'what model',
            'how do you work', 'what are you', 'who created you'
        ]
        if any(indicator in message_lower for indicator in general_chat_indicators):
            return {'academic': True, 'recommendation': False}
        
        # RULE 2: Check for course/learning questions (not college recommendations)
        course_learning_indicators = [
            'recommend me courses', 'suggest courses', 'what courses',
            'which course should i', 'courses to study', 'learn abroad',
            'study abroad courses', 'what should i study'
        ]
        if any(indicator in message_lower for indicator in course_learning_indicators):
            return {'academic': True, 'recommendation': False}
        
        # RULE 3: College recommendation indicators - NOW INCLUDES VAGUE REQUESTS
        college_rec_indicators = [
            'recommend me colleges', 'suggest me colleges', 'recommend colleges',
            'suggest colleges', 'recommend me some colleges', 'suggest some colleges',
            'top engineering colleges', 'best colleges for', 'colleges in',
            'engineering colleges in', 'medical colleges in', 'list of colleges',
            'colleges offering', 'recommend college for', 'suggest university',
            'which college should', 'show me colleges', 'good colleges',
            'best colleges', 'college suggestions'
        ]
        if any(indicator in message_lower for indicator in college_rec_indicators):
            return {'academic': False, 'recommendation': True}
        
        # RULE 4: Check for greetings
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(message_lower.startswith(greeting) for greeting in greetings):
            return {'academic': True, 'recommendation': False}
        
        # Default to academic for everything else
        return {'academic': True, 'recommendation': False}

class DynamicAcademicChatbot:
    def __init__(self, openai_api_key: str, excel_path: str, db_path: str, model_name: str = "gpt-3.5-turbo"):
        """Initialize the dynamic chatbot that recommends without asking for preferences"""
        
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
        self.db_manager = DatabaseManager(db_path)
        self.college_data_manager = CollegeDataManager(excel_path)
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
        self._setup_preference_extraction()
    
    def _setup_academic_chain(self):
        """Setup academic conversation chain"""
        academic_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Alex, a friendly and knowledgeable academic assistant. You help with:

WHAT YOU DO:
- Answer general conversation questions naturally
- Explain academic subjects and concepts
- Provide study strategies and learning advice
- Offer general career guidance and educational pathway advice
- Discuss courses and learning resources
- Help with exam preparation strategies
- Answer questions about the chatbot itself
- Engage in friendly, appropriate general conversation

IMPORTANT:
- For college recommendations: Let the recommendation system handle it
- Be helpful and informative across all academic topics

Personality: Warm, helpful, conversational, and knowledgeable across many topics."""),
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
        """Setup college recommendation chain that works with minimal info"""
        recommendation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized college admission counselor. Your role is to provide college recommendations based on available information.

When you have specific criteria (location, course, etc.):
- Provide specific, relevant college recommendations
- Explain why these colleges match the criteria
- Give practical information about admission processes

When information is minimal/vague:
- Provide general, well-known college recommendations
- Suggest popular options across different categories
- Mention that more specific preferences can help narrow down options
- Be encouraging and helpful

Always:
- Reference the specific colleges found in recommendations
- Be factual and helpful
- Provide actionable information"""),
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
    
    def _setup_preference_extraction(self):
        """Setup preference extraction"""
        self.preference_parser = PydanticOutputParser(pydantic_object=UserPreferences)
        self.preference_prompt = PromptTemplate(
            template="""
            Extract user preferences for college search from the conversation.
            
            Conversation History:
            {conversation_history}
            
            Current Message:
            {current_message}
            
            IMPORTANT: Extract whatever preferences you can find, even if minimal.
            If no specific preferences are mentioned, return null values but still extract any available information.
            
            {format_instructions}
            
            Extract preferences as JSON.
            """,
            input_variables=["conversation_history", "current_message"],
            partial_variables={"format_instructions": self.preference_parser.get_format_instructions()}
        )
        
        self.preference_chain = LLMChain(llm=self.academic_llm, prompt=self.preference_prompt)
    
    def extract_preferences_with_llm(self, chat_id: str, current_message: str) -> UserPreferences:
        """Extract user preferences using LLM - now accepts minimal info"""
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
            return UserPreferences()  # Return empty preferences - we'll work with what we have
    
    def get_openai_college_recommendations(self, preferences: UserPreferences, location: str = None) -> List[Dict]:
        """Get college recommendations from OpenAI - works with minimal preferences"""
        try:
            pref_parts = []
            
            # Build preference description - include whatever we have
            if preferences.specific_institution_type:
                pref_parts.append(f"Institution type: {preferences.specific_institution_type}")
            if location or preferences.location:
                loc = location or preferences.location
                pref_parts.append(f"Location: {loc}")
            if preferences.course_type:
                pref_parts.append(f"Course type: {preferences.course_type}")
            if preferences.specific_course:
                pref_parts.append(f"Specific course: {preferences.specific_course}")
            if preferences.college_type:
                pref_parts.append(f"College type: {preferences.college_type}")
            if preferences.level:
                pref_parts.append(f"Level: {preferences.level}")
            
            # If we have some preferences, use them. Otherwise get general recommendations.
            if pref_parts:
                preference_text = ", ".join(pref_parts)
                prompt = f"""Based on these criteria: {preference_text}

Recommend exactly 5 well-known colleges/universities in India that match these preferences.

Provide response as a JSON array:
[
    {{
        "id": "unique_id",
        "name": "College Name",
        "location": "City, State",
        "type": "Government/Private/Deemed",
        "courses_offered": "Main courses offered",
        "website": "Official website",
        "admission_process": "Admission process",
        "approximate_fees": "Fee range per year",
        "notable_features": "Key highlights",
        "source": "openai_knowledge"
    }}
]

Return ONLY the JSON array, nothing else."""
            else:
                # General recommendations for vague requests
                prompt = """Recommend exactly 5 well-known and diverse colleges/universities in India that are popular among students.
Include a mix of government and private institutions across different regions.

Provide response as a JSON array:
[
    {{
        "id": "unique_id",
        "name": "College Name",
        "location": "City, State",
        "type": "Government/Private/Deemed",
        "courses_offered": "Main courses offered",
        "website": "Official website",
        "admission_process": "Admission process",
        "approximate_fees": "Fee range per year",
        "notable_features": "Key highlights",
        "source": "openai_knowledge"
    }}
]

Return ONLY the JSON array, nothing else."""
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
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
        """Format college recommendations to ALWAYS return exactly 5 colleges"""
        recommendations = []
        database_count = 0
        openai_count = 0
        
        # Process database colleges FIRST
        logger.info(f"Processing {len(filtered_colleges)} database colleges...")
        for item in filtered_colleges:
            if len(recommendations) >= 5:
                break
            college = item['college']
            json_rec = self.convert_database_college_to_json(college, item['score'], item['reasons'])
            if json_rec:
                recommendations.append(json_rec)
                database_count += 1
        
        # Add OpenAI colleges to reach exactly 5
        if len(recommendations) < 5 and openai_colleges:
            needed = 5 - len(recommendations)
            logger.info(f"Adding {needed} OpenAI colleges to reach 5 total...")
            for college in openai_colleges[:needed]:
                json_rec = self.convert_openai_college_to_json(college)
                if json_rec:
                    recommendations.append(json_rec)
                    openai_count += 1
        
        # If still less than 5, get more from OpenAI
        if len(recommendations) < 5:
            logger.info(f"Still need {5 - len(recommendations)} more colleges, getting additional from OpenAI...")
            additional_colleges = self.get_openai_college_recommendations(preferences)
            for college in additional_colleges:
                if len(recommendations) >= 5:
                    break
                # Avoid duplicates
                if college['name'] not in [r['name'] for r in recommendations]:
                    json_rec = self.convert_openai_college_to_json(college)
                    if json_rec:
                        recommendations.append(json_rec)
                        openai_count += 1
        
        # Create detailed text summary
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
            # Fallback if no colleges found
            text_summary = "Here are some well-known colleges you might consider:"
        
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
        """Main processing function - NOW PROVIDES RECOMMENDATIONS WITHOUT ASKING"""
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
        
        # Detect query intent - REMOVED "needs_more_info" logic
        intent = self.intent_detector.detect_intent(message)
        logger.info(f"Query: '{message}' | Detected intent: {intent}")
        
        # Process through chains based on intent
        academic_response = ""
        recommendation_response = ""
        recommendations_data = []
        is_recommendation = False
        
        # Academic chain processing
        if intent['academic']:
            try:
                logger.info("Processing through Academic Chain...")
                academic_response = self.academic_chain.invoke({
                    "input": message,
                    "chat_id": chat_id
                })
                # Save to academic memory
                self.chat_memories[chat_id].save_context(
                    {"input": message},
                    {"output": academic_response}
                )
                logger.info("✅ Academic chain completed successfully")
            except Exception as e:
                logger.error(f"❌ Academic chain error: {e}")
                academic_response = "I'm having trouble processing your request. Please try again."
        
        # Recommendation chain processing - NOW ALWAYS PROVIDES RECOMMENDATIONS
        if intent['recommendation']:
            try:
                logger.info("Processing through Recommendation Chain...")
                
                # Extract preferences (can be minimal/empty)
                preferences = self.extract_preferences_with_llm(chat_id, message)
                logger.info(f"Extracted preferences: {preferences}")
                
                # Get college data from database
                filtered_colleges = self.college_data_manager.filter_colleges_by_preferences(preferences)
                logger.info(f"Database colleges found: {len(filtered_colleges)}")
                
                # Get OpenAI recommendations (works with minimal preferences)
                logger.info("Getting OpenAI recommendations...")
                openai_colleges = self.get_openai_college_recommendations(preferences, preferences.location)
                logger.info(f"OpenAI colleges found: {len(openai_colleges)}")
                
                # Format recommendations (ALWAYS aim for exactly 5)
                recommendations_data, summary_text = self.format_college_recommendations(
                    filtered_colleges, openai_colleges, preferences
                )
                
                is_recommendation = len(recommendations_data) > 0
                logger.info(f"Total recommendations prepared: {len(recommendations_data)}")
                
                # Generate recommendation response that works with any level of detail
                recommendation_input = f"User is asking for college recommendations: {message}"
                if recommendations_data:
                    if any([preferences.location, preferences.course_type, preferences.specific_course]):
                        # Specific recommendations
                        recommendation_input += f"\n\nI have found {len(recommendations_data)} colleges matching the mentioned criteria. Please provide a helpful response introducing these options."
                    else:
                        # General recommendations for vague requests
                        recommendation_input += f"\n\nSince the user didn't specify detailed preferences, I've provided {len(recommendations_data)} well-known colleges across different categories. Please introduce these general recommendations and mention that more specific preferences can help narrow down options."
                
                recommendation_response = self.recommendation_chain.invoke({
                    "input": recommendation_input,
                    "chat_id": chat_id
                })
                
                # Save to recommendation memory
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
        
        # Create final response
        final_response = self._create_final_response(
            academic_response, recommendation_response, message, intent
        )
        
        # Save final response
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
        """Create final response based on intent"""
        
        # Handle error cases
        if not academic_response and not recommendation_response:
            return "I apologize, but I encountered an error. Please try again!"
        
        # Single intent responses
        if intent['academic'] and not intent['recommendation']:
            return academic_response or "I'm here to help! Please ask me anything."
        
        if intent['recommendation'] and not intent['academic']:
            return recommendation_response or "Here are some college recommendations for you!"
        
        # Fallback
        return academic_response or recommendation_response or "I'm here to help! Please ask me about academics or college recommendations."

# Initialize environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "chatbot.db")
EXCEL_PATH = os.getenv("EXCEL_PATH", "colleges.xlsx")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")
    OPENAI_API_KEY = "your-openai-api-key-here"

# Initialize the dynamic chatbot
try:
    chatbot = DynamicAcademicChatbot(OPENAI_API_KEY, EXCEL_PATH, DB_PATH)
    logger.info("✅ Dynamic Academic Chatbot initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing chatbot: {e}")
    raise

# FastAPI Routes
@app.get("/")
async def root():
    return {
        "message": "Dynamic Academic Chatbot API - Provides recommendations without asking for preferences",
        "version": "4.1.0",
        "features": [
            "✅ Always provides college recommendations",
            "✅ Works with minimal or no preference information",
            "✅ No explicit preference asking",
            "✅ Dynamic intent detection",
            "✅ Always returns exactly 5 colleges"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, chat_id: str = Query(..., description="Chat ID managed by backend")):
    """Dynamic chat endpoint that provides recommendations without asking for preferences"""
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
            "service": "Dynamic Academic Chatbot API",
            "version": "4.1.0",
            "database": "connected",
            "college_data": f"{college_count} colleges loaded",
            "features": {
                "dynamic_recommendations": "✅",
                "no_preference_asking": "✅",
                "minimal_info_handling": "✅",
                "always_5_colleges": "✅"
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
    
    logger.info("🚀 Starting Dynamic Academic Chatbot API...")
    logger.info("🎯 Version 4.1.0 - Provides recommendations without asking for preferences")
    logger.info("🔗 API: http://localhost:8000")
    logger.info("📚 Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
