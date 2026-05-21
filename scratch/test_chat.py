import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

try:
    print("Initializing ChatGoogleGenerativeAI...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key
    )
    print("Invoking model...")
    res = llm.invoke("Hi, who are you?")
    print(f"Success! Response: {res.content}")
except Exception as e:
    print(f"Error: {e}")
