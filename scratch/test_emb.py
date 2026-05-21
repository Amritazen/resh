import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

try:
    print("Initializing GoogleGenerativeAIEmbeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    print("Embedding query...")
    res = embeddings.embed_query("Hello world")
    print(f"Success! Vector length: {len(res)}")
except Exception as e:
    print(f"Error: {e}")
