import streamlit as st
import os
import time
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

# Set up environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Custom color scheme and styling
custom_css = """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap');
    
    /* Professional navy theme */
    :root {
        --primary-color: #0f172a;
        --secondary-color: #2563eb;
        --accent-color: #38bdf8;
        --text-color: #e2e8f0;
        --muted-text: #94a3b8;
        --user-bg: #1d4ed8;
        --assistant-bg: #1e293b;
    }
    
    /* Page styling - removing background color overrides */
    .stApp {
        background: #0b1120;
        color: var(--text-color);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #172554 0%, #1e3a8a 55%, #0369a1 100%);
        color: #ffffff;
        padding: 2.5rem;
        border: 1px solid rgba(125, 211, 252, 0.28);
        border-radius: 18px;
        box-shadow: 0 12px 32px rgba(2, 6, 23, 0.35);
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .header-text h1, .header-text p {
        color: #ffffff !important;
    }
    
    .header-image {
        flex-shrink: 0;
    }
    
    .header-image img {
        max-width: 300px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    

    
    .dancing-script {
        font-size: 58px;
        font-family: 'Dancing Script', cursive !important;
    }
    
    /* Chat container styling */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Message styling */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin: 0.8rem 0;
        border: 1px solid #334155;
        background: var(--assistant-bg);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.16);
        color: #e2e8f0;
    }

    /* Blue bubbles for the user's queries */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        border-color: #60a5fa;
        margin-left: 10%;
    }

    /* Slate bubbles for the assistant's answers */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: var(--assistant-bg);
        border-color: #334155;
        margin-right: 10%;
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] span {
        color: #f8fafc !important;
    }
    
    /* Input box styling */
    div[data-testid="stChatInput"] {
        border: 1px solid #334155;
        border-radius: 14px;
        background: #111827;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }

    div[data-testid="stChatInput"] textarea {
        color: #f8fafc !important;
    }

    div[data-testid="stChatInput"] button {
        background: #2563eb;
        color: white;
        border-radius: 9px;
    }
    
    /* Button styling */
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: #ffffff;
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
        border: 1px solid #60a5fa;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
    }
    
    /* Clear Chat button specific styling */
    .clear-chat-button {
        background-color: #1e293b !important;
        color: #cbd5e1 !important;
        border-radius: 15px !important;
        padding: 0.3rem 1rem !important;
        font-size: 0.8rem !important;
        border: 1px solid #475569 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease !important;
    }
    
    .clear-chat-button:hover {
        background-color: #334155 !important;
        border-color: #64748b !important;
        transform: translateY(-1px) !important;
    }
    
    /* Status indicator styling */
    div[data-testid="stStatusWidget"] {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Hide default elements */
    #MainMenu, footer, .stDeployButton, #stDecoration {display: none;}
    button[title="View fullscreen"] {display: none;}
    
    /* Legal disclaimer styling */
    .legal-disclaimer {
        background-color: rgba(30, 41, 59, 0.85);
        border: 1px solid #475569;
        border-left: 4px solid #f59e0b;
        padding: 1.15rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .warning-message {
        background-color: rgba(146, 64, 14, 0.22);
        color: #fef3c7;
        padding: 10px;
        border-radius: 4px;
        margin: 5px 0;
        border: 1px solid #f59e0b;
    }
</style>
"""

# Page configuration
st.set_page_config(
    page_title="Better Call Bot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Header section
try:
    image_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "saul.png"
    )

    if os.path.exists(image_path):
        st.markdown("""
            <div class="header-container">
                <div class="header-text">
                    <h1>
                        <span class="dancing-script">Better Call Bot!</span>
                    </h1>
                    <p>
                        Did you know that you have rights?
                        The Constitution says you do. And so do I.
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([4, 2, 4])

        with col2:
            st.image(
                Image.open(image_path),
                width=220
            )

    else:
        st.warning("Saul image not found.")

except Exception as e:
    st.error(f"Unable to load image: {e}")
    
    
# Add disclaimer before chat interface
disclaimer_text = """
<div class="legal-disclaimer" style="color:#ffffff;" >
    <h4>⚠️ Legal Information Disclaimer</h4>
    <p>This chatbot provides general legal information, NOT legal advice. The information provided:</p>
    <ul>
        <li>Is for informational purposes only</li>
        <li>Is not a substitute for professional legal counsel</li>
        <li>May not be up-to-date or applicable to your jurisdiction</li>
        <li>Should not be relied upon for making legal decisions</li>
    </ul>
    <p><strong>Please consult with a qualified attorney for specific legal advice.</strong></p>
</div>
"""

st.markdown(disclaimer_text, unsafe_allow_html=True)

# Reset conversation function
def reset_conversation():
    st.session_state.messages = []
    st.session_state.chat_history.clear()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(BASE_DIR, "saul.png")

VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

db_retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Define the prompt template
prompt_template = """
<s>[INST]You are a legal information chatbot with strict limitations. Follow these guidelines:

1. NEVER provide specific legal advice
2. If the question seeks specific legal advice or involves complex legal matters, respond with a warning to seek professional legal counsel
3. Only provide publicly available legal information with proper citations
4. Use clear qualifying language (e.g., "generally," "typically," "it may depend")
5. If unsure, explicitly state the limitations of the information
6. For questions about:
   - Ongoing legal proceedings: Decline to comment
   - Specific legal strategy: Refer to an attorney
   - Complex legal interpretation: Emphasize need for professional counsel
7. **CRITICAL RULE: NEVER mention or cite non-Indian laws, organizations, or legal bodies (e.g., ABA, NHTSA). Only refer to Indian law and Indian legal bodies.**

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")	
# Helper function to format chat history
def format_chat_history():
    history_text = ""
    messages = st.session_state.chat_history.messages
    for msg in messages[-4:]:  # Last 2 exchanges (4 messages)
        if isinstance(msg, HumanMessage):
            history_text += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"
    return history_text

# Chat interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages with improved styling
for message in st.session_state.messages:
    with st.chat_message(
        message.get("role"),
        avatar="👤" if message.get("role") == "user" else "⚖️"
    ):
        content = message.get("content")
        # Split content into main response and sources if sources exist
        if "Sources:" in content:
            main_content, sources = content.split("Sources:", 1)
            st.write(main_content)
            st.markdown("**Sources:**" + sources)
        else:
            st.write(content)

# Function to check for risky content
def check_for_risky_content(response):
    risky_keywords = ['you should', 'I advise', 'you must', 'definitely', 'always', 'never']
    return any(keyword in response.lower() for keyword in risky_keywords)

# Chat input with custom styling
input_prompt = st.chat_input("Ask your legal question...")

if input_prompt:
    with st.chat_message("user", avatar="👤"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})
    st.session_state.chat_history.add_user_message(input_prompt)

    with st.chat_message("assistant", avatar="⚖️"):
        with st.status("Analyzing your question...", expanded=True):
            # Retrieve relevant documents
            docs = db_retriever.invoke(input_prompt)
            
            # Format context from documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format chat history
            chat_history_text = format_chat_history()
            
            # Create the prompt
            formatted_prompt = prompt.format(
                context=context,
                question=input_prompt,
                chat_history=chat_history_text
            )
            
            # Get response from LLM
            response = llm.invoke(formatted_prompt)
            response_text = response.content
            
            # Check for risky content
            if check_for_risky_content(response_text):
                st.markdown("""
                    <div class="warning-message">
                        ⚠️ This response may contain general guidance. Please consult with a qualified attorney for specific advice.
                    </div>
                """, unsafe_allow_html=True)
        
            
            # Display response with typing effect
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response_text:
                full_response += chunk
                time.sleep(0.02)
                message_placeholder.markdown(full_response + " ▌")
            message_placeholder.markdown(full_response)
            
        col1, col2, col3 = st.columns([4, 1, 4])
        with col2:
            st.button('🗑️ Clear Chat', on_click=reset_conversation, key="clear_chat", help="Clear the conversation history", type="secondary", use_container_width=True)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.chat_history.add_ai_message(response_text)

st.markdown('</div>', unsafe_allow_html=True)
