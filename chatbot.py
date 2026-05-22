import os
from dotenv import load_dotenv
 
load_dotenv()
 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
 
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key,
    )
 
def get_rag_answer(question, chroma_path):
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "GOOGLE_API_KEY not found. Please check your config."
 
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
 
        if not os.path.exists(chroma_path):
            return "No documents uploaded yet. Please upload a file first."
 
        db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        results = db.similarity_search(question, k=5)
 
        if not results:
            return "I could not find any relevant information in the uploaded document."
 
        context = "\n---\n".join([res.page_content for res in results])
 
        prompt_template = """You are Resh, a helpful document assistant.
Answer the question based ONLY on the context provided below.
If the answer is not found in the context, say: "I could not find that in the uploaded document."
Be clear and detailed in your answer.
 
Context:
{context}
 
Question: {question}
 
Answer:"""
 
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        final_prompt = prompt.format(context=context, question=question)
 
        llm = get_llm()
        response = llm.invoke(final_prompt)
        return response.content if hasattr(response, 'content') else str(response)
 
    except Exception as e:
        err = str(e).lower()
        print(f"RAG ERROR: {e}")
        if "quota" in err or "429" in err or "resource_exhausted" in err:
            return "Resh is busy right now. Please try again in a moment."
        if "api_key" in err or "invalid" in err:
            return "Configuration error: invalid API key. Please contact the site owner."
        return "Something went wrong. Please try again in a moment."
 
def get_direct_answer(question):
    try:
        llm = get_llm()
        prompt_template = """You are Resh, a helpful AI assistant.
Answer the following question clearly using your general knowledge.
 
Question: {question}
 
Answer:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
        final_prompt = prompt.format(question=question)
        response = llm.invoke(final_prompt)
        return response.content if hasattr(response, 'content') else str(response)
 
    except Exception as e:
        err = str(e).lower()
        print(f"CHAT ERROR: {e}")
        if "quota" in err or "429" in err or "resource_exhausted" in err:
            return "Resh is busy right now. Please try again in a moment."
        if "api_key" in err or "invalid" in err:
            return "Configuration error: invalid API key. Please contact the site owner."
        return "Something went wrong. Please try again in a moment."
