import os
import sys
import io
from dotenv import load_dotenv
 
# Force UTF-8 at top
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ["PYTHONIOENCODING"] = "utf-8"
 
load_dotenv()
 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
 
def extract_text(response):
    """Safely extracts text from Gemini response using multiple fallbacks"""
    try:
        content = getattr(response, 'content', response)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get('text', ''))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts).strip()
        if hasattr(response, 'text'):
            return response.text
        return str(content).strip()
    except Exception:
        return str(response)
 
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment. Please check your .env file.")
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.3,
        google_api_key=api_key,
        convert_system_message_to_human=True
    )
 
# ✅ KEY FIX: chroma_path is now a parameter, not a global constant
def get_rag_answer(question, chroma_path):
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "GOOGLE_API_KEY not found in environment. Please check your config."
 
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
 
        # ✅ Check THIS user's ChromaDB, not a shared one
        if not os.path.exists(chroma_path):
            return "No documents uploaded yet. Please upload a file first."
 
        db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
 
        results = db.similarity_search(question, k=4)
 
        if not results:
            return "I could not find any relevant information in the uploaded document."
 
        context = "\n---\n".join([res.page_content for res in results])
 
        prompt_template = """You are Resh. Answer ONLY from the context.
If user asks about skills return ONLY skills.
If user asks about projects return ONLY projects.
Never mix sections. If not found say:
I could not find that in the uploaded document.
 
Context: {context}
 
Question: {question}
 
Answer:"""
 
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        final_prompt = prompt.format(context=context, question=question)
 
        llm = get_llm()
        response = llm.invoke(final_prompt)
        answer = extract_text(response)
        return str(answer)
 
    except Exception as e:
        err = str(e).lower()
        print(f"CHATBOT ERROR: {str(e)}")
        if "quota" in err or "429" in err or "resource_exhausted" in err:
            return "Resh is busy right now — the AI service is at capacity. Please try again in a moment."
        if "api_key" in err or "api key" in err or "invalid" in err:
            return "Configuration error: invalid API key. Please contact the site owner."
        if "not found" in err or "model" in err:
            return "AI model error. Please try again."
        return "Something went wrong. Please try again in a moment."
 
def get_direct_answer(question):
    try:
        llm = get_llm()
        prompt_template = """You are Resh, a helpful assistant.
Answer clearly from general knowledge.
 
Question: {question}
 
Answer:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
        final_prompt = prompt.format(question=question)
        response = llm.invoke(final_prompt)
        answer = extract_text(response)
        return str(answer)
 
    except Exception as e:
        err = str(e).lower()
        print(f"CHATBOT ERROR: {str(e)}")
        if "quota" in err or "429" in err or "resource_exhausted" in err:
            return "Resh is busy right now — the AI service is at capacity. Please try again in a moment."
        if "api_key" in err or "api key" in err or "invalid" in err:
            return "Configuration error: invalid API key. Please contact the site owner."
        if "not found" in err or "model" in err:
            return "AI model error. Please try again."
        return "Something went wrong. Please try again in a moment."
