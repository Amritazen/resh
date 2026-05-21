import os
import sys
import traceback
# At very top before everything
os.environ["PYTHONIOENCODING"] = "utf-8"

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from chatbot import get_rag_answer, get_direct_answer
from ingest import ingest_single_file

app = Flask(__name__)
CORS(app)

# Directory setup
UPLOAD_FOLDER = "uploads"
DATA_FOLDER = "data"
CHROMA_PATH = "chroma_db"

for folder in [UPLOAD_FOLDER, DATA_FOLDER, CHROMA_PATH]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'txt', 'docx', 'pptx'}

@app.route("/")
def landing():
    """Landing page"""
    return render_template("landing.html")

@app.route("/ask")
def index():
    """Chatbot 'Ask anything' page"""
    return render_template("index.html")

@app.route("/file-chat")
def file_chat_page():
    """Chat with file page"""
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
            answer = get_rag_answer(message)
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
            original_filename = secure_filename(file.filename)
            filename = original_filename
            # Ensure unique filename to avoid overwrite
            counter = 1
            while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
                name, ext = os.path.splitext(original_filename)
                filename = f"{name}_{counter}{ext}"
                counter += 1
            # Save to upload folder (relative path) and get absolute path for ingestion
            rel_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(rel_path)
            abs_path = os.path.abspath(rel_path)
            # Ingest the file and handle any exceptions
            try:
                ingest_single_file(abs_path)
            except Exception as e:
                traceback.print_exc()
                return jsonify({"success": False, "error": f"Ingestion failed: {str(e)}"})

            return jsonify({"success": True, "filename": filename})
        else:
            return jsonify({"success": False, "error": "File type not supported."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Upload error: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
