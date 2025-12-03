# ---------------------------------------------
# 1️⃣ REQUIRED LIBRARIES (Imports)
# ---------------------------------------------
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Environment variables load kar rahe hain (.env file se)
load_dotenv()

# API Key check kar rahe hain
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ Error: Please .env file mein apni GOOGLE_API_KEY daalein!")
    exit()

# ---------------------------------------------
# 2️⃣ LOAD YOUR DATA (CSV)
# ---------------------------------------------
print("📊 Data load ho raha hai...")
# CSV file padh rahe hain
df = pd.read_csv("titanic.csv")
print("✅ Data loaded successfully!")
print(df.head()) # Thoda sa data dikhayega

# ---------------------------------------------
# 3️⃣ SETUP GEMINI LLM (BRAIN)
# ---------------------------------------------
print("🤖 Gemini Brain initialize ho raha hai...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,           # 0 matlab accurate answers, creative nahi
    google_api_key=api_key
)

# ---------------------------------------------
# 4️⃣ CREATE AGENT (MAGIC)
# ---------------------------------------------
# Ye agent Pandas DataFrame ke saath baat kar sakta hai
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,            # True matlab background mein kya soch raha hai wo dikhayega
    allow_dangerous_code=True # Code run karne ki permission
)
print("✨ Agent ready hai!")

# ---------------------------------------------
# 5️⃣ ASK QUESTIONS (CHAT LOOP)
# ---------------------------------------------
print("\n💬 Chat shuru karein! (Type 'exit' to stop)")

while True:
    # User se sawal puch rahe hain
    user_q = input("\nAsk anything about your data: ")
    
    # Agar user 'exit' ya 'quit' likhe toh band kar do
    if user_q.lower() in ["exit", "quit"]:
        print("👋 Bye Bye!")
        break
    
    try:
        # Agent se jawab maang rahe hain
        response = agent.invoke(user_q)
        
        # Jawab print kar rahe hain
        # Note: Agent ka output dictionary ho sakta hai, isliye hum 'output' key dhundenge
        if isinstance(response, dict) and 'output' in response:
            print("\nAnswer:", response['output'])
        else:
            print("\nAnswer:", response)
            
    except Exception as e:
        print(f"\n❌ Kuch gadbad hui: {e}")
