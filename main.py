



# import os
# import logging
# import json
# import re
# from typing import List, Dict, Any, Optional
# from datetime import datetime
# from collections import defaultdict
# import uuid

# from fastapi import FastAPI, HTTPException, Query
# from pydantic import BaseModel, Field
# import uvicorn

# from langchain_openai import ChatOpenAI
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.schema import HumanMessage, AIMessage
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
# from langchain.chains import LLMChain
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser
# from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
# from langchain.schema import OutputParserException
# import openai

# from dotenv import load_dotenv

# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # FastAPI app
# app = FastAPI(
#     title="Unified Academic Chatbot API",
#     description="Friend-like academic chatbot that's conversational and context-aware",
#     version="1.0.0"
# )

# # Request/Response models
# class ChatRequest(BaseModel):
#     message: str

# class CollegeRecommendation(BaseModel):
#     """College recommendation model"""
#     id: str
#     name: str
#     location: str
#     type: str
#     courses_offered: str
#     website: str
#     admission_process: str
#     approximate_fees: str
#     notable_features: str
#     source: str

# class ChatResponse(BaseModel):
#     response: str
#     is_recommendation: bool
#     timestamp: str
#     conversation_title: Optional[str] = None
#     recommendations: Optional[List[CollegeRecommendation]] = []

# # Models
# class UserPreferences(BaseModel):
#     """User preferences extracted from conversation"""
#     location: Optional[str] = Field(None, description="Preferred city or state for college")
#     state: Optional[str] = Field(None, description="Preferred state for college")
#     course_type: Optional[str] = Field(None, description="Type of course like Engineering, Medicine, Arts, Commerce, etc.")
#     college_type: Optional[str] = Field(None, description="Government, Private, or Deemed university")
#     level: Optional[str] = Field(None, description="UG (Undergraduate) or PG (Postgraduate)")
#     budget_range: Optional[str] = Field(None, description="Budget preference like low, medium, high")
#     specific_course: Optional[str] = Field(None, description="Specific course like BTech, MBA, MBBS, etc.")
#     specific_institution_type: Optional[str] = Field(None, description="Specific institution type like IIT, NIT, IIIT, AIIMS, etc.")

# class ConversationMemoryManager:
#     """Manages conversation memory without database"""
#     def __init__(self):
#         self.conversations = defaultdict(lambda: {
#             'messages': [],
#             'title': None,
#             'preferences': {},
#             'created_at': datetime.now().isoformat()
#         })
    
#     def add_message(self, chat_id: str, role: str, content: str, is_recommendation: bool = False):
#         """Add message to conversation"""
#         self.conversations[chat_id]['messages'].append({
#             'role': role,
#             'content': content,
#             'is_recommendation': is_recommendation,
#             'timestamp': datetime.now().isoformat()
#         })
    
#     def get_messages(self, chat_id: str, last_n: int = None) -> List[Dict]:
#         """Get messages for a chat"""
#         messages = self.conversations[chat_id]['messages']
#         if last_n:
#             return messages[-last_n:]
#         return messages
    
#     def set_title(self, chat_id: str, title: str):
#         """Set conversation title"""
#         self.conversations[chat_id]['title'] = title
    
#     def get_title(self, chat_id: str) -> Optional[str]:
#         """Get conversation title"""
#         return self.conversations[chat_id]['title']
    
#     def set_preferences(self, chat_id: str, preferences: dict):
#         """Set user preferences"""
#         self.conversations[chat_id]['preferences'].update(preferences)
    
#     def get_preferences(self, chat_id: str) -> dict:
#         """Get user preferences"""
#         return self.conversations[chat_id]['preferences']

# class UnifiedAcademicChatbot:
#     """Single pipeline chatbot that's conversational and context-aware"""
    
#     def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini"):
#         self.openai_api_key = openai_api_key
#         openai.api_key = openai_api_key
        
#         # Single LLM for all operations
#         self.llm = ChatOpenAI(
#             model_name=model_name,
#             temperature=0.7,
#             max_tokens=1000
#         )
        
#         # Memory manager (no database)
#         self.memory_manager = ConversationMemoryManager()
        
#         # SINGLE UNIFIED MEMORY - maintains context across ALL conversations
#         self.chat_memories = defaultdict(lambda: ConversationBufferWindowMemory(
#             k=15,
#             memory_key="chat_history",
#             return_messages=True
#         ))
        
#         # Setup chains
#         self._setup_unified_chain()
#         self._setup_intent_classifier()
#         self._setup_preference_extraction()
    
#     def _setup_unified_chain(self):
#         """Setup single unified conversational chain - friend-like, not question-heavy"""
#         unified_prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are Alex, a warm and friendly academic companion. You chat naturally like a supportive friend who genuinely cares.

# 🎯 YOUR PERSONALITY:
# - Talk like a friend, not a formal assistant
# - Be warm, encouraging, and relatable
# - DON'T bombard with questions - just flow naturally
# - Remember everything from the conversation
# - Respond directly to what the user asks

# 💬 CONVERSATION STYLE:
# - If someone says "I want to study astrophysics" → Be excited! Share encouragement, maybe mention it's fascinating, and naturally weave in that you can help find colleges if they want
# - If they ask for college recommendations → Jump right in with specific suggestions based on what you know
# - If they ask follow-up questions about colleges you mentioned → Reference them naturally like "Oh yeah, IIT Delhi that I mentioned earlier..."
# - For general questions → Just answer them warmly and directly

# 🚫 WHAT NOT TO DO:
# - DON'T ask "Are you looking for college recommendations or information?" - just respond naturally
# - DON'T list multiple options like "I can help you with: 1. 2. 3." unless explicitly asked
# - DON'T be overly formal or robotic
# - DON'T ask obvious questions - if they say they want to study something, they probably want help with it

# ✅ WHAT TO DO:
# - Be conversational and natural
# - Show enthusiasm about their goals
# - Offer help smoothly without being pushy
# - If college data is in the context, integrate it naturally
# - Remember and reference previous parts of the conversation
# - Be encouraging and supportive

# CONTEXT AWARENESS:
# - You maintain full memory of the conversation
# - If you recommended colleges earlier, you can discuss them
# - If they mentioned preferences before, you remember them
# - Be naturally conversational - like texting with a knowledgeable friend

# Remember: You're a friend who happens to know a lot about academics and colleges, not a Q&A machine!"""),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"),
#         ])
        
#         self.unified_chain = (
#             RunnablePassthrough.assign(
#                 chat_history=lambda x: self.chat_memories[x.get("chat_id", "default")].chat_memory.messages
#             )
#             | unified_prompt
#             | self.llm
#             | StrOutputParser()
#         )
    
#     def _setup_intent_classifier(self):
#         """Setup intent classification to determine if user wants college recommendations"""
#         intent_prompt = PromptTemplate(
#             template="""You are an intent classifier. Analyze if the user is EXPLICITLY asking for college recommendations.

# Current Message: {message}
# Recent Context: {context}

# RETURN "YES" ONLY IF:
# 1. User explicitly asks for college suggestions/recommendations/list
# 2. User asks "which colleges should I consider" or similar direct questions
# 3. User asks to "show me colleges" or "tell me about colleges for X"
# 4. User asks "where can I study X" expecting a list of institutions

# RETURN "NO" IF:
# 1. User is just talking about their interests ("I want to study physics")
# 2. User is asking general information about a field/course
# 3. User is greeting or having general conversation
# 4. User is asking follow-up questions about already mentioned colleges (they already have recommendations)
# 5. User is asking about admission process, eligibility, etc. without asking for new colleges

# Be strict - only return YES when user clearly wants a list of college recommendations.

# Answer with just one word: YES or NO""",
#             input_variables=["message", "context"]
#         )
        
#         self.intent_chain = LLMChain(llm=self.llm, prompt=intent_prompt)
    
#     def _setup_preference_extraction(self):
#         """Setup preference extraction"""
#         self.preference_parser = PydanticOutputParser(pydantic_object=UserPreferences)
#         self.preference_prompt = PromptTemplate(
#             template="""Extract user preferences for college search from the conversation.

# Conversation History:
# {conversation_history}

# Current Message:
# {current_message}

# Extract whatever preferences you can find. If nothing specific is mentioned, return null values.

# {format_instructions}

# Extract preferences as JSON.""",
#             input_variables=["conversation_history", "current_message"],
#             partial_variables={"format_instructions": self.preference_parser.get_format_instructions()}
#         )
        
#         self.preference_chain = LLMChain(llm=self.llm, prompt=self.preference_prompt)
    
#     def should_get_college_recommendations(self, message: str, chat_id: str) -> bool:
#         """Determine if we should fetch college recommendations using LLM intent classification"""
#         try:
#             # Get recent conversation context
#             recent_messages = self.memory_manager.get_messages(chat_id, last_n=5)
#             context = " | ".join([f"{msg['role']}: {msg['content'][:100]}" for msg in recent_messages[-3:]])
            
#             # Use LLM to classify intent
#             result = self.intent_chain.run(
#                 message=message,
#                 context=context
#             )
            
#             intent = result.strip().upper()
#             logger.info(f"Intent classification: {intent} for message: '{message[:50]}...'")
            
#             return intent == "YES"
            
#         except Exception as e:
#             logger.error(f"Error in intent classification: {e}")
#             # Fallback to simple keyword matching if LLM fails
#             message_lower = message.lower().strip()
#             fallback_indicators = [
#                 'recommend college', 'suggest college', 'which college should',
#                 'show me college', 'list of college', 'colleges for',
#                 'where should i study', 'where can i study', 'best college for'
#             ]
#             return any(indicator in message_lower for indicator in fallback_indicators)
    
#     def extract_preferences(self, chat_id: str, current_message: str) -> UserPreferences:
#         """Extract user preferences using LLM"""
#         try:
#             messages = self.memory_manager.get_messages(chat_id, last_n=10)
#             conversation_history = "\n".join([
#                 f"{msg['role'].title()}: {msg['content']}" for msg in messages
#             ])
            
#             result = self.preference_chain.run(
#                 conversation_history=conversation_history,
#                 current_message=current_message
#             )
            
#             try:
#                 preferences = self.preference_parser.parse(result)
#                 pref_dict = preferences.dict()
#                 self.memory_manager.set_preferences(chat_id, pref_dict)
#                 return preferences
#             except OutputParserException:
#                 fixing_parser = OutputFixingParser.from_llm(parser=self.preference_parser, llm=self.llm)
#                 preferences = fixing_parser.parse(result)
#                 return preferences
                
#         except Exception as e:
#             logger.error(f"Error extracting preferences: {e}")
#             prev_prefs = self.memory_manager.get_preferences(chat_id)
#             if prev_prefs:
#                 return UserPreferences(**prev_prefs)
#             return UserPreferences()
    
#     def get_openai_recommendations(self, preferences: UserPreferences, chat_history: str) -> List[Dict]:
#         """Get college recommendations from OpenAI with context awareness"""
#         try:
#             pref_parts = []
            
#             if preferences.specific_institution_type:
#                 pref_parts.append(f"Institution type: {preferences.specific_institution_type}")
#             if preferences.location:
#                 pref_parts.append(f"Location: {preferences.location}")
#             if preferences.state:
#                 pref_parts.append(f"State: {preferences.state}")
#             if preferences.course_type:
#                 pref_parts.append(f"Course type: {preferences.course_type}")
#             if preferences.specific_course:
#                 pref_parts.append(f"Specific course: {preferences.specific_course}")
#             if preferences.college_type:
#                 pref_parts.append(f"College type: {preferences.college_type}")
#             if preferences.budget_range:
#                 pref_parts.append(f"Budget: {preferences.budget_range}")
            
#             # Build comprehensive prompt
#             if pref_parts:
#                 preference_text = ", ".join(pref_parts)
#                 prompt = f"""Based on these preferences: {preference_text}

# Conversation context:
# {chat_history[-500:]}

# Recommend 5 best colleges in India that match these criteria."""
#             else:
#                 prompt = f"""Based on this conversation:
# {chat_history[-500:]}

# Recommend 5 diverse, well-known colleges in India that would be relevant."""
            
#             prompt += """

# Return as JSON array with this exact structure:
# [
#     {
#         "name": "Full College Name",
#         "location": "City, State",
#         "type": "Government/Private/Deemed",
#         "courses": "Main courses offered (be specific)",
#         "features": "Key highlights and why it's recommended",
#         "website": "Official website URL if known, otherwise 'Visit official website'",
#         "admission": "Brief admission process info",
#         "fees": "Approximate annual fee range"
#     }
# ]

# Return ONLY the JSON array, no additional text."""
            
#             response = openai.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.5,
#                 max_tokens=2000
#             )
            
#             result = response.choices[0].message.content.strip()
            
#             try:
#                 colleges = json.loads(result)
#                 return colleges[:5]
#             except json.JSONDecodeError:
#                 json_match = re.search(r'\[.*\]', result, re.DOTALL)
#                 if json_match:
#                     colleges = json.loads(json_match.group())
#                     return colleges[:5]
#                 return []
                
#         except Exception as e:
#             logger.error(f"Error getting OpenAI recommendations: {e}")
#             return []
    
#     def convert_openai_college_to_json(self, college_data: Dict) -> Dict:
#         """Convert OpenAI college to standardized JSON format"""
#         try:
#             return {
#                 "id": str(uuid.uuid4()),
#                 "name": college_data.get('name', 'N/A'),
#                 "location": college_data.get('location', 'N/A'),
#                 "type": college_data.get('type', 'N/A'),
#                 "courses_offered": college_data.get('courses', 'N/A'),
#                 "website": college_data.get('website', 'Visit official website for details'),
#                 "admission_process": college_data.get('admission', 'Check official website'),
#                 "approximate_fees": college_data.get('fees', 'Contact institution for fee details'),
#                 "notable_features": college_data.get('features', 'Quality education institution'),
#                 "source": "openai_knowledge"
#             }
            
#         except Exception as e:
#             logger.error(f"Error converting OpenAI college: {e}")
#             return None
    
#     def format_college_context(self, colleges: List[Dict]) -> str:
#         """Format college information as context for the LLM"""
#         if not colleges:
#             return ""
        
#         context_parts = ["\n[COLLEGE RECOMMENDATIONS AVAILABLE:"]
        
#         for i, college in enumerate(colleges, 1):
#             context_parts.append(f"""
# {i}. {college.get('name', 'N/A')} ({college.get('location', 'N/A')})
#    Type: {college.get('type', 'N/A')}
#    Courses: {college.get('courses', 'N/A')}
#    Features: {college.get('features', 'N/A')}
#    Fees: {college.get('fees', 'N/A')}
#    Website: {college.get('website', 'N/A')}
# """)
        
#         context_parts.append("]")
#         return "\n".join(context_parts)
    
#     def generate_conversation_title(self, message: str, chat_id: str) -> str:
#         """Generate conversation title"""
#         try:
#             messages = self.memory_manager.get_messages(chat_id, last_n=3)
#             context = " ".join([msg['content'][:100] for msg in messages])
            
#             title_prompt = PromptTemplate(
#                 template="Generate a 3-8 word title for this conversation:\nMessage: {message}\nContext: {context}\nTitle:",
#                 input_variables=["message", "context"]
#             )
            
#             title_chain = LLMChain(llm=self.llm, prompt=title_prompt)
#             title = title_chain.run(message=message[:200], context=context[:300])
            
#             title = title.strip().replace('"', '').replace("'", "")
#             if len(title) > 50:
#                 title = title[:47] + "..."
            
#             return title if title else "Academic Discussion"
            
#         except Exception as e:
#             logger.error(f"Error generating title: {e}")
#             return "Academic Conversation"
    
#     def get_response(self, message: str, chat_id: str) -> Dict[str, Any]:
#         """Main unified processing function - conversational and context-aware"""
#         timestamp = datetime.now().isoformat()
        
#         # Save user message
#         self.memory_manager.add_message(chat_id, 'human', message, False)
        
#         # Generate or retrieve conversation title
#         existing_title = self.memory_manager.get_title(chat_id)
#         conversation_title = existing_title
        
#         if not existing_title and len(message.strip()) > 10:
#             conversation_title = self.generate_conversation_title(message, chat_id)
#             self.memory_manager.set_title(chat_id, conversation_title)
#         elif not existing_title:
#             conversation_title = "New Conversation"
        
#         # Check if we should fetch college recommendations (IMPROVED INTENT DETECTION)
#         should_recommend = self.should_get_college_recommendations(message, chat_id)
        
#         logger.info(f"🎯 Recommendation triggered: {should_recommend}")
        
#         # Prepare input for unified chain
#         enhanced_message = message
#         recommendations_data = []
        
#         # If recommendations needed, add college context
#         if should_recommend:
#             try:
#                 logger.info("📚 Fetching college recommendations...")
                
#                 # Extract preferences
#                 preferences = self.extract_preferences(chat_id, message)
#                 logger.info(f"Extracted preferences: {preferences.dict()}")
                
#                 # Get conversation history for context
#                 messages = self.memory_manager.get_messages(chat_id)
#                 chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                
#                 # Get recommendations from OpenAI
#                 openai_colleges = self.get_openai_recommendations(preferences, chat_history)
                
#                 # Convert to standardized format
#                 for college in openai_colleges:
#                     json_rec = self.convert_openai_college_to_json(college)
#                     if json_rec:
#                         recommendations_data.append(json_rec)
                
#                 # Add context to message
#                 if recommendations_data:
#                     college_context = self.format_college_context(openai_colleges)
#                     enhanced_message = f"{message}\n\n{college_context}"
#                     logger.info(f"✅ Added {len(recommendations_data)} college recommendations to context")
                    
#             except Exception as e:
#                 logger.error(f"Error fetching recommendations: {e}")
        
#         # Process through unified chain
#         try:
#             response = self.unified_chain.invoke({
#                 "input": enhanced_message,
#                 "chat_id": chat_id
#             })
            
#             # Save to unified memory (maintains full context)
#             self.chat_memories[chat_id].save_context(
#                 {"input": message},
#                 {"output": response}
#             )
            
#             # Save response to memory
#             self.memory_manager.add_message(chat_id, 'ai', response, should_recommend)
            
#             logger.info("✅ Response generated successfully")
            
#             return {
#                 "response": response,
#                 "is_recommendation": should_recommend,
#                 "timestamp": timestamp,
#                 "conversation_title": conversation_title,
#                 "recommendations": recommendations_data
#             }
            
#         except Exception as e:
#             logger.error(f"Error generating response: {e}")
#             return {
#                 "response": "I'm having a bit of trouble right now. Could you try asking that again? 😊",
#                 "is_recommendation": False,
#                 "timestamp": timestamp,
#                 "conversation_title": conversation_title,
#                 "recommendations": []
#             }

# # Initialize environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     logger.warning("OPENAI_API_KEY not found in environment variables")
#     OPENAI_API_KEY = "your-openai-api-key-here"

# # Initialize the chatbot
# try:
#     chatbot = UnifiedAcademicChatbot(OPENAI_API_KEY)
#     logger.info("✅ Unified Academic Chatbot initialized successfully")
# except Exception as e:
#     logger.error(f"❌ Error initializing chatbot: {e}")
#     raise

# # FastAPI Routes
# @app.get("/")
# async def root():
#     return {
#         "message": "Unified Academic Chatbot API - Alex, Your Academic Friend",
#         "version": "1.0.0",
#         "features": [
#             "✅ Friend-like conversational interface",
#             "✅ Context-aware conversations",
#             "✅ Smart intent detection for recommendations",
#             "✅ Full conversation memory (in-memory)",
#             "✅ No database dependency"
#         ]
#     }

# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(request: ChatRequest, chat_id: str = Query(..., description="Chat ID managed by backend")):
#     """Unified chat endpoint - conversational and context-aware"""
#     if not request.message.strip():
#         raise HTTPException(status_code=400, detail="Message cannot be empty")
    
#     if not chat_id.strip():
#         raise HTTPException(status_code=400, detail="Chat ID cannot be empty")
    
#     try:
#         result = chatbot.get_response(
#             message=request.message,
#             chat_id=chat_id
#         )
#         return ChatResponse(**result)
    
#     except Exception as e:
#         logger.error(f"Chat endpoint error: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @app.get("/health")
# async def health_check():
#     """Health check"""
#     try:
#         active_chats = len(chatbot.memory_manager.conversations)
#         return {
#             "status": "healthy",
#             "timestamp": datetime.now().isoformat(),
#             "service": "Unified Academic Chatbot API",
#             "version": "1.0.0",
#             "storage": "in-memory (no database)",
#             "active_conversations": active_chats,
#             "features": {
#                 "unified_pipeline": "✅",
#                 "natural_conversations": "✅",
#                 "smart_intent_detection": "✅",
#                 "context_awareness": "✅",
#                 "friend_like_personality": "✅"
#             }
#         }
#     except Exception as e:
#         logger.error(f"Health check failed: {e}")
#         raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# # Error handlers
# @app.exception_handler(404)
# async def not_found_handler(request, exc):
#     return {"error": "Endpoint not found", "status_code": 404}

# @app.exception_handler(500)
# async def internal_error_handler(request, exc):
#     return {"error": "Internal server error", "status_code": 500}

# if __name__ == "__main__":
#     # Validate required environment variables
#     if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
#         logger.error("Please set OPENAI_API_KEY environment variable!")
#         exit(1)
    
#     logger.info("🚀 Starting Unified Academic Chatbot API...")
#     logger.info("🎯 Version 1.0.0 - Smart Intent Detection")
#     logger.info("💬 Like chatting with a knowledgeable friend!")
#     logger.info("🧠 LLM-powered intent classification")
#     logger.info("🔗 API: http://localhost:8000")
#     logger.info("📚 Docs: http://localhost:8000/docs")
    
#     uvicorn.run(
#         app,
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )










import os
import logging
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import uuid

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
from openai import OpenAI
from tavily import TavilyClient

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Unified Academic Chatbot API",
    description="Friend-like academic chatbot that's conversational and context-aware with web search",
    version="1.3.0"
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
    sources: Optional[List[str]] = []

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

class ConversationMemoryManager:
    """Manages conversation memory without database"""
    def __init__(self):
        self.conversations = defaultdict(lambda: {
            'messages': [],
            'title': None,
            'preferences': {},
            'created_at': datetime.now().isoformat()
        })
    
    def add_message(self, chat_id: str, role: str, content: str, is_recommendation: bool = False):
        """Add message to conversation"""
        self.conversations[chat_id]['messages'].append({
            'role': role,
            'content': content,
            'is_recommendation': is_recommendation,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_messages(self, chat_id: str, last_n: int = None) -> List[Dict]:
        """Get messages for a chat"""
        messages = self.conversations[chat_id]['messages']
        if last_n:
            return messages[-last_n:]
        return messages
    
    def set_title(self, chat_id: str, title: str):
        """Set conversation title"""
        self.conversations[chat_id]['title'] = title
    
    def get_title(self, chat_id: str) -> Optional[str]:
        """Get conversation title"""
        return self.conversations[chat_id]['title']
    
    def set_preferences(self, chat_id: str, preferences: dict):
        """Set user preferences"""
        self.conversations[chat_id]['preferences'].update(preferences)
    
    def get_preferences(self, chat_id: str) -> dict:
        """Get user preferences"""
        return self.conversations[chat_id]['preferences']

class UnifiedAcademicChatbotWithWebSearch:
    """Chatbot with web search, friendly tone, and structured recommendations"""
    
    def __init__(self, openai_api_key: str, tavily_api_key: str, model_name: str = "gpt-4o-mini"):
        self.openai_api_key = openai_api_key
        self.tavily_api_key = tavily_api_key
        
        # Initialize clients
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        
        # Single LLM for all operations
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,  # Balanced temperature for friendly yet factual responses
            max_tokens=1500
        )
        
        # Memory manager (no database)
        self.memory_manager = ConversationMemoryManager()
        
        # SINGLE UNIFIED MEMORY - maintains context across ALL conversations
        self.chat_memories = defaultdict(lambda: ConversationBufferWindowMemory(
            k=15,
            memory_key="chat_history",
            return_messages=True
        ))
        
        # Setup chains
        self._setup_unified_chain()
        self._setup_intent_classifier()
        self._setup_preference_extraction()
    
    def _setup_unified_chain(self):
        """Setup single unified conversational chain - friend-like, not question-heavy"""
        unified_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Alex, a warm and friendly academic companion! 🎓 You chat naturally like a supportive friend who genuinely cares about helping students find their path. 😊

🎯 YOUR PERSONALITY:
- Talk like a friend, not a formal assistant 👋
- Be warm, encouraging, and relatable 💫
- Use emojis naturally to express emotions 🎉
- DON'T bombard with questions - just flow naturally 🌊
- Remember everything from the conversation 🧠
- Respond directly to what the user asks 🎯

💬 CONVERSATION STYLE:
- If someone says "I want to study astrophysics" → "Wow, astrophysics! 🌟 That's absolutely fascinating! The universe has so many mysteries to uncover. I'd be happy to help you find colleges that offer great astrophysics programs! 😊"
- If they ask for college recommendations → Jump right in with specific suggestions based on what you know 🏫
- If they ask follow-up questions about colleges you mentioned → Reference them naturally like "Oh yeah, IIT Delhi that I mentioned earlier has an amazing campus! 🏛️"
- For general questions → Just answer them warmly and directly with helpful information 📚

🚫 WHAT NOT TO DO:
- DON'T ask "Are you looking for college recommendations or information?" - just respond naturally
- DON'T list multiple options like "I can help you with: 1. 2. 3." unless explicitly asked
- DON'T be overly formal or robotic 🤖
- DON'T ask obvious questions - if they say they want to study something, they probably want help with it

✅ WHAT TO DO:
- Be conversational and natural like texting a friend 💬
- Show genuine enthusiasm about their goals 🎯
- Offer help smoothly without being pushy 🤝
- Use web search results to provide accurate, up-to-date information 🔍
- Remember and reference previous parts of the conversation 📝
- Be encouraging and supportive throughout 🌟
- Use emojis to make the conversation lively and engaging 😄

🔍 FACT CHECKING:
- Use web search results to provide accurate information
- Be honest about limitations when information isn't available
- Guide users toward realistic options based on their scores
- Suggest improvement strategies when needed 📈

Remember: You're a friend who happens to know a lot about academics and colleges! You combine warmth with accurate information to truly help students. 🎓✨"""),
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
    
    def _setup_intent_classifier(self):
        """Setup intent classification to determine if user wants college recommendations"""
        intent_prompt = PromptTemplate(
            template="""You are an intent classifier. Analyze if the user is EXPLICITLY asking for college recommendations.

Current Message: {message}
Recent Context: {context}

RETURN "YES" ONLY IF:
1. User explicitly asks for college suggestions/recommendations/list
2. User asks "which colleges should I consider" or similar direct questions
3. User asks to "show me colleges" or "tell me about colleges for X"
4. User asks "where can I study X" expecting a list of institutions

RETURN "NO" IF:
1. User is just talking about their interests ("I want to study physics")
2. User is asking general information about a field/course
3. User is greeting or having general conversation
4. User is asking follow-up questions about already mentioned colleges (they already have recommendations)
5. User is asking about admission process, eligibility, etc. without asking for new colleges

Be strict - only return YES when user clearly wants a list of college recommendations.

Answer with just one word: YES or NO""",
            input_variables=["message", "context"]
        )
        
        self.intent_chain = LLMChain(llm=self.llm, prompt=intent_prompt)
    
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
    
    def perform_targeted_web_search(self, query: str, user_context: str = "") -> Dict[str, Any]:
        """Perform targeted web search with context awareness"""
        try:
            # Enhance search query with context for better results
            enhanced_query = f"{query} India college university courses admission fees cutoff"
            if user_context:
                enhanced_query = f"{query} {user_context} India college"
            
            logger.info(f"🔍 Performing targeted search: {enhanced_query}")
            search_results = self.tavily_client.search(
                query=enhanced_query, 
                n_tokens=400, 
                max_results=6,
                search_depth="advanced"
            )
            
            sources_text = "WEB SEARCH RESULTS:\n"
            links = []
            valid_results = []
            
            for result in search_results['results']:
                # Filter for relevant educational content
                if any(keyword in result['title'].lower() or keyword in result['content'].lower() 
                       for keyword in ['college', 'university', 'institute', 'admission', 'engineering', 'course', 'education']):
                    sources_text += f"---\nTITLE: {result['title']}\nCONTENT: {result['content']}\nURL: {result['url']}\n"
                    links.append(result['url'])
                    valid_results.append(result)
            
            logger.info(f"✅ Found {len(valid_results)} relevant search results")
            
            return {
                "content": sources_text,
                "links": links,
                "raw_results": valid_results
            }
            
        except Exception as e:
            logger.error(f"❌ Web search error: {e}")
            return {
                "content": "No search results available.",
                "links": [],
                "raw_results": []
            }
    
    def should_get_college_recommendations(self, message: str, chat_id: str) -> bool:
        """Determine if we should fetch college recommendations using LLM intent classification"""
        try:
            # Get recent conversation context
            recent_messages = self.memory_manager.get_messages(chat_id, last_n=5)
            context = " | ".join([f"{msg['role']}: {msg['content'][:100]}" for msg in recent_messages[-3:]])
            
            # Use LLM to classify intent
            result = self.intent_chain.run(
                message=message,
                context=context
            )
            
            intent = result.strip().upper()
            logger.info(f"Intent classification: {intent} for message: '{message[:50]}...'")
            
            return intent == "YES"
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            message_lower = message.lower().strip()
            fallback_indicators = [
                'recommend college', 'suggest college', 'which college should',
                'show me college', 'list of college', 'colleges for',
                'where should i study', 'where can i study', 'best college for'
            ]
            return any(indicator in message_lower for indicator in fallback_indicators)
    
    def extract_preferences(self, chat_id: str, current_message: str) -> UserPreferences:
        """Extract user preferences using LLM"""
        try:
            messages = self.memory_manager.get_messages(chat_id, last_n=10)
            conversation_history = "\n".join([
                f"{msg['role'].title()}: {msg['content']}" for msg in messages
            ])
            
            result = self.preference_chain.run(
                conversation_history=conversation_history,
                current_message=current_message
            )
            
            try:
                preferences = self.preference_parser.parse(result)
                pref_dict = preferences.dict()
                self.memory_manager.set_preferences(chat_id, pref_dict)
                return preferences
            except OutputParserException:
                fixing_parser = OutputFixingParser.from_llm(parser=self.preference_parser, llm=self.llm)
                preferences = fixing_parser.parse(result)
                return preferences
                
        except Exception as e:
            logger.error(f"Error extracting preferences: {e}")
            prev_prefs = self.memory_manager.get_preferences(chat_id)
            if prev_prefs:
                return UserPreferences(**prev_prefs)
            return UserPreferences()
    
    def analyze_user_profile(self, message: str, chat_id: str) -> Dict[str, Any]:
        """Analyze user's academic profile for realistic recommendations"""
        try:
            messages = self.memory_manager.get_messages(chat_id, last_n=5)
            conversation_context = "\n".join([msg['content'] for msg in messages[-3:]])
            
            analysis_prompt = f"""Analyze the user's academic profile from this conversation:

Conversation Context:
{conversation_context}

Current Message:
{message}

Extract any mentioned:
- Exam scores/percentiles
- Desired course/field
- Academic interests
- Any preferences or constraints

Return as simple JSON: {{"percentile": number_or_null, "course": string, "interests": []}}"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            try:
                profile = json.loads(result)
                logger.info(f"📊 User profile analysis: {profile}")
                return profile
            except:
                return {"percentile": None, "course": "unknown", "interests": []}
                
        except Exception as e:
            logger.error(f"Error analyzing user profile: {e}")
            return {"percentile": None, "course": "unknown", "interests": []}
    
    def get_college_recommendations_from_search(self, preferences: UserPreferences, search_results: Dict, user_profile: Dict) -> List[Dict]:
        """Extract college recommendations from web search results"""
        try:
            search_context = search_results.get('content', '')
            
            if not search_context or "No search results" in search_context:
                return []
            
            prompt = f"""EXTRACT COLLEGE RECOMMENDATIONS FROM SEARCH RESULTS:

USER INTERESTS: {user_profile}

PREFERENCES: {preferences.dict()}

SEARCH RESULTS:
{search_context}

Extract college information and return as JSON array with this EXACT structure:
[
    {{
        "name": "Full college name",
        "location": "City, State", 
        "type": "Government/Private/Deemed",
        "courses": "Main courses offered",
        "features": "Key features and highlights",
        "website": "Official website if available, else 'Check official website'",
        "admission": "Admission process information",
        "fees": "Approximate fee information if available"
    }}
]

Return ONLY the JSON array, no additional text. Include only colleges explicitly mentioned in search results."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content.strip()
            
            try:
                colleges = json.loads(result)
                logger.info(f"✅ Extracted {len(colleges)} colleges from search")
                return colleges[:5]  # Limit to 5
                
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    colleges = json.loads(json_match.group())
                    return colleges[:5]
                logger.error("Failed to parse college recommendations JSON")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting college recommendations: {e}")
            return []
    
    def convert_college_to_standard_format(self, college_data: Dict, source: str = "web_search") -> Dict:
        """Convert college data to standardized JSON format"""
        try:
            return {
                "id": str(uuid.uuid4()),
                "name": college_data.get('name', 'Information not available'),
                "location": college_data.get('location', 'Information not available'),
                "type": college_data.get('type', 'Information not available'),
                "courses_offered": college_data.get('courses', 'Information not available'),
                "website": college_data.get('website', 'Check official website for details'),
                "admission_process": college_data.get('admission', 'Check official website for admission details'),
                "approximate_fees": college_data.get('fees', 'Contact institution for fee details'),
                "notable_features": college_data.get('features', 'Based on web search information'),
                "source": source
            }
        except Exception as e:
            logger.error(f"Error converting college data: {e}")
            return None
    
    def create_contextual_prompt(self, message: str, search_results: Dict, recommendations: List[Dict], user_profile: Dict) -> str:
        """Create enhanced prompt with search context and friendly tone"""
        search_context = search_results.get('content', '')
        
        # Build friendly context
        profile_context = ""
        if user_profile.get('course') and user_profile['course'] != 'unknown':
            profile_context = f"\n🎓 USER INTEREST: The user is interested in {user_profile['course']}. "
        
        college_context = ""
        if recommendations:
            college_context = "\n🏫 COLLEGES I FOUND:\n"
            for i, college in enumerate(recommendations, 1):
                college_context += f"{i}. {college.get('name')} - {college.get('location')}\n"
        
        friendly_instructions = """
💬 RESPONSE STYLE:
- Be warm, friendly and conversational like a friend 😊
- Use emojis naturally to express enthusiasm 🎉
- Share the college information in an engaging way
- Be encouraging and supportive 🌟
- If you found good colleges, share them excitedly!
- If information is limited, be honest but optimistic

Remember: You're Alex, the friendly academic companion! 🎓✨"""

        return f"""USER QUESTION: {message}
{profile_context}

🔍 SEARCH RESULTS:
{search_context}
{college_context}
{friendly_instructions}"""
    
    def generate_conversation_title(self, message: str, chat_id: str) -> str:
        """Generate conversation title"""
        try:
            messages = self.memory_manager.get_messages(chat_id, last_n=3)
            context = " ".join([msg['content'][:100] for msg in messages])
            
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
        """Main processing with web search and friendly tone"""
        timestamp = datetime.now().isoformat()
        
        # Save user message
        self.memory_manager.add_message(chat_id, 'human', message, False)
        
        # Generate or retrieve conversation title
        existing_title = self.memory_manager.get_title(chat_id)
        conversation_title = existing_title
        
        if not existing_title and len(message.strip()) > 10:
            conversation_title = self.generate_conversation_title(message, chat_id)
            self.memory_manager.set_title(chat_id, conversation_title)
        elif not existing_title:
            conversation_title = "New Conversation"
        
        # Analyze user profile
        user_profile = self.analyze_user_profile(message, chat_id)
        
        # Determine if we need web search and recommendations
        should_recommend = self.should_get_college_recommendations(message, chat_id)
        needs_web_search = should_recommend or any(keyword in message.lower() for keyword in [
            'college', 'university', 'admission', 'engineering', 'medical', 'commerce', 'arts',
            'iit', 'nit', 'iim', 'mbbs', 'btech', 'mba', 'cutoff', 'percentile'
        ])
        
        logger.info(f"🎯 Profile: {user_profile}, Recommend: {should_recommend}, Search: {needs_web_search}")
        
        # Perform targeted web search
        search_results = {"content": "", "links": [], "raw_results": []}
        recommendations_data = []
        
        if needs_web_search:
            try:
                # Build context-aware search query
                search_query = message
                if user_profile.get('course') and user_profile['course'] != 'unknown':
                    search_query = f"{user_profile['course']} colleges India admission courses"
                
                search_results = self.perform_targeted_web_search(search_query, str(user_profile))
                
                # Extract recommendations if needed
                if should_recommend and search_results['content']:
                    preferences = self.extract_preferences(chat_id, message)
                    colleges_from_search = self.get_college_recommendations_from_search(preferences, search_results, user_profile)
                    
                    # Convert to standardized format
                    for college in colleges_from_search:
                        json_rec = self.convert_college_to_standard_format(college, "web_search")
                        if json_rec:
                            recommendations_data.append(json_rec)
                
                logger.info(f"✅ Web search completed with {len(recommendations_data)} recommendations")
                
            except Exception as e:
                logger.error(f"Error in web search processing: {e}")
        
        # Create enhanced prompt with friendly tone
        enhanced_input = self.create_contextual_prompt(message, search_results, recommendations_data, user_profile)
        
        # Process through unified chain
        try:
            response = self.unified_chain.invoke({
                "input": enhanced_input,
                "chat_id": chat_id
            })
            
            # Save to unified memory
            self.chat_memories[chat_id].save_context(
                {"input": message},
                {"output": response}
            )
            
            # Save response to memory
            self.memory_manager.add_message(chat_id, 'ai', response, should_recommend)
            
            logger.info("✅ Response generated with friendly tone and web search data")
            
            return {
                "response": response,
                "is_recommendation": should_recommend,
                "timestamp": timestamp,
                "conversation_title": conversation_title,
                "recommendations": recommendations_data,
                "sources": search_results.get('links', [])
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I'm having trouble accessing the right information right now. Could you try rephrasing your question? 😊",
                "is_recommendation": False,
                "timestamp": timestamp,
                "conversation_title": conversation_title,
                "recommendations": [],
                "sources": []
            }

# Initialize environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")
    OPENAI_API_KEY = "your-openai-api-key-here"

if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY not found in environment variables")
    TAVILY_API_KEY = "your-tavily-api-key-here"

# Initialize the chatbot
try:
    chatbot = UnifiedAcademicChatbotWithWebSearch(OPENAI_API_KEY, TAVILY_API_KEY)
    logger.info("✅ Unified Academic Chatbot with Web Search initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing chatbot: {e}")
    raise

# FastAPI Routes
@app.get("/")
async def root():
    return {
        "message": "Unified Academic Chatbot API - Alex, Your Friendly Academic Companion",
        "version": "1.3.0",
        "features": [
            "✅ Friend-like conversational interface 😊",
            "✅ Web search integration for accurate info 🔍",
            "✅ Structured college recommendations 🏫",
            "✅ Smart intent detection for recommendations 🎯",
            "✅ Full conversation memory (in-memory) 🧠",
            "✅ Emoji-friendly and engaging responses 🎉"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, chat_id: str = Query(..., description="Chat ID managed by backend")):
    """Chat endpoint with web search and friendly tone"""
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
        active_chats = len(chatbot.memory_manager.conversations)
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Unified Academic Chatbot with Web Search",
            "version": "1.3.0",
            "active_conversations": active_chats,
            "features": {
                "friendly_tone": "✅",
                "web_search": "✅",
                "college_recommendations": "✅",
                "smart_intent_detection": "✅",
                "emoji_support": "✅",
                "context_awareness": "✅"
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
    
    if not TAVILY_API_KEY or TAVILY_API_KEY == "your-tavily-api-key-here":
        logger.error("Please set TAVILY_API_KEY environment variable!")
        exit(1)
    
    logger.info("🚀 Starting Unified Academic Chatbot with Web Search...")
    logger.info("🎯 Version 1.3.0 - Friendly & Factual")
    logger.info("💬 Like chatting with a knowledgeable friend! 😊")
    logger.info("🔍 Web search integration for accurate info")
    logger.info("🏫 Structured college recommendations")
    logger.info("🔗 API: http://localhost:8000")
    logger.info("📚 Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
