# Import necessary libraries
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from a .env file (if available)
load_dotenv()

# Define prompt template for AI model interaction
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries."),  # System message guiding AI behavior
        ("user", "Question: {question}")  # User's input formatted for AI processing
    ]
)

# Set Streamlit page configuration (title, icon, layout)
st.set_page_config(page_title="Llama3 AI Chat", page_icon="🤖", layout="centered")

# Apply custom styling using CSS to improve UI appearance
st.markdown(
    """
    <style>
        .main { background-color: #f8f9fa; } /* Light gray background */
        .stTextArea textarea { font-size: 16px; } /* Increase font size for better readability */
        .stButton button { width: 100%; background-color: #4CAF50; color: white; } /* Style button */
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with app information and instructions
with st.sidebar:
    st.header("ℹ️ About This App")  # Sidebar title
    st.write(
        """
        - **Model:** Llama3:2:1B  # Model version being used
        - **Built with:** LangChain & Streamlit  
        - **How to use:**  
          1. Type a question in the input box.  
          2. Click 'Generate Response'.  
          3. Wait for the AI to respond!  
        """
    )

# Main app title and subtitle
st.title("🤖 Llama3 AI Chat")
st.subheader("Ask me anything!")

# Text input box for user queries (allows multi-line input)
input_text = st.text_area("Enter your question below:", height=150)

# Initialize the Llama3 model
llm = Ollama(model="llama3.2:1b")

# Define output parser to process AI responses
output_parser = StrOutputParser()

# Create processing chain that combines prompt, model, and output parsing
chain = prompt | llm | output_parser

# Button to trigger AI response generation
if st.button("🔍 Generate Response"):
    if input_text.strip():  # Check if user entered a question
        with st.spinner("Generating response... ⏳"):  # Show loading indicator
            response = chain.invoke({'question': input_text})  # Generate AI response
            st.markdown(f"**🗨️ AI Response:**\n\n{response}")  # Display AI response
    else:
        st.warning("⚠️ Please enter a question before generating a response.")  # Warning for empty input
