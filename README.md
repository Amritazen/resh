# Resh — Real-time Embedded Smart Helper

Resh is a professional, white-labeled RAG (Retrieval-Augmented Generation) chatbot designed for general queries and deep document analysis. Powered by **Google Gemini 3 Flash Preview**.

## 🚀 Quick Start (Local)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Create a `.env` file and add your Google API Key:
   ```text
   GOOGLE_API_KEY=your_api_key_here
   ```

3. **Run the App**:
   ```bash
   python app.py
   ```
   Access at: `http://localhost:8080`

## 🐳 Docker Deployment

To run Resh using Docker:

1. **Build the image**:
   ```bash
   docker build -t resh-app .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8080:8080 --env-file .env resh-app
   ```

## 🛠 Features

- **General Chat**: Ask anything using global knowledge.
- **File Chat**: Upload PDF, DOCX, TXT, or PPTX files for focused Q&A.
- **Smart Retrieval**: Uses ChromaDB and Sentence Transformers for precise context matching.
- **Responsive Design**: Optimized for both Desktop and Mobile views.

## 📁 Project Structure

- `app.py`: Main Flask application.
- `chatbot.py`: AI logic and model integration.
- `ingest.py`: Document processing and vector storage.
- `templates/`: Professional HTML interfaces (Landing, Ask, File Chat).
- `data/` & `uploads/`: Document storage (excluded from Git).
- `chroma_db/`: Vector database (excluded from Git).

---
*Built with transparency and performance in mind.*
