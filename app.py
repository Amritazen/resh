import os
import sys
import uuid
import traceback

os.environ["PYTHONIOENCODING"] = "utf-8"

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from chatbot import get_rag_answer, get_direct_answer
from ingest import ingest_single_file

app = Flask(__name__)
CORS(app)

# SECRET KEY — required for session to work
app.secret_key = os.environ.get("SECRET_KEY", "resh-secret-key-2024")

# Directory setup
UPLOAD_FOLDER = "uploads"
DATA_FOLDER = "data"
CHROMA_BASE_PATH = "chroma_sessions"  # Each user gets a subfolder here

for folder in [UPLOAD_FOLDER, DATA_FOLDER, CHROMA_BASE_PATH]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'txt', 'docx', 'pptx'}

def get_session_id():
    """Get or create a unique session ID for this user."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_chroma_path(session_id):
    """Get the ChromaDB path for a specific session."""
    return os.path.join(CHROMA_BASE_PATH, session_id)

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/ask")
def index():
    return render_template("index.html")

@app.route("/file-chat")
def file_chat_page():
    return render_template("file_chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        message = data.get("message", "").strip()
        mode = data.get("mode", "direct")

        if not message:
            return jsonify({"response": "Please type something."})

        if mode == "upload":
            session_id = get_session_id()
            chroma_path = get_chroma_path(session_id)
            answer = get_rag_answer(message, chroma_path)
        else:
            answer = get_direct_answer(message)

        return jsonify({"response": str(answer)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"response": f"Error: {str(e)}"})

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part in request."})

        file = request.files['file']

        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file."})

        if file and allowed_file(file.filename):
            session_id = get_session_id()

            original_filename = secure_filename(file.filename)
            filename = f"{session_id}_{original_filename}"

            rel_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(rel_path)
            abs_path = os.path.abspath(rel_path)

            chroma_path = get_chroma_path(session_id)

            try:
                ingest_single_file(abs_path, chroma_path)
            except Exception as e:
                traceback.print_exc()
                return jsonify({"success": False, "error": f"Ingestion failed: {str(e)}"})

            return jsonify({"success": True, "filename": original_filename})
        else:
            return jsonify({"success": False, "error": "File type not supported."})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Upload error: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
