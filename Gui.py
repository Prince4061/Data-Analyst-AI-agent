# ---------------------------------------------
# 1️⃣ REQUIRED LIBRARIES (Imports)
# ---------------------------------------------
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader

# Environment variables load kar rahe hain
load_dotenv()

# ---------------------------------------------
# 2️⃣ PAGE CONFIGURATION (Title & Layout)
# ---------------------------------------------
st.set_page_config(page_title="Data Analyst AI", page_icon="📊")

st.title("📊 Data Analyst AI Agent")
st.write("Build By Tarun Kaushik")

# Sidebar for API Key (Optional, agar .env mein na ho)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter Google API Key", type="password")

if not api_key:
    st.warning("⚠️ Please enter your Google API Key to proceed.")
    st.stop()

# ---------------------------------------------
# 3️⃣ FILE UPLOADER (CSV & PDF)
# ---------------------------------------------
uploaded_file = st.file_uploader("Choose a file", type=["csv", "pdf"])

if uploaded_file is not None:
    # -----------------------------------------
    # CASE A: CSV FILE
    # -----------------------------------------
    if uploaded_file.name.endswith(".csv"):
        st.success("✅ CSV File Uploaded!")
        
        # CSV ko DataFrame mein convert kar rahe hain
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Data:")
        st.dataframe(df.head())

        # Agent Setup
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key
        )
        
        # Agent bana rahe hain
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True
        )

        # Chat Interface
        user_question = st.text_input("Ask a question about your CSV:")
        
        if user_question:
            with st.spinner("Thinking... 🤔"):
                try:
                    response = agent.invoke(user_question)
                    st.write("### 🤖 Answer:")
                    if isinstance(response, dict) and 'output' in response:
                        st.write(response['output'])
                    else:
                        st.write(response)
                except Exception as e:
                    st.error(f"Error: {e}")

    # -----------------------------------------
    # CASE B: PDF FILE (Simple Text Read)
    # -----------------------------------------
    elif uploaded_file.name.endswith(".pdf"):
        st.success("✅ PDF File Uploaded!")
        
        # PDF se text nikaal rahe hain
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        st.write("### PDF Content Preview (First 500 chars):")
        st.text(text[:500] + "...")
        
        # Note: PDF ke liye hum simple LLM call use karenge (Agent nahi)
        # Kyunki DataFrame Agent sirf Tables/CSV ke liye hota hai.
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=api_key
        )
        
        user_question = st.text_input("Ask a question about your PDF:")
        
        if user_question:
            with st.spinner("Reading PDF & Thinking... 🤔"):
                # Simple Prompt: Context + Question
                prompt = f"Context:\n{text}\n\nQuestion: {user_question}\nAnswer:"
                response = llm.invoke(prompt)
                st.write("### 🤖 Answer:")
                st.write(response.content)

else:
    st.info("👆 Please upload a CSV or PDF file to start.")
