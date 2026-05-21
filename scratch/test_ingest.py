import os, sys
sys.path.append('c:/Users/SMILE/OneDrive/Desktop/resh')
from ingest import ingest_single_file
from chatbot import get_rag_answer

# Ensure upload folder exists
upload_folder = 'uploads'
os.makedirs(upload_folder, exist_ok=True)

sample_path = os.path.join(upload_folder, 'sample.txt')
with open(sample_path, 'w', encoding='utf-8') as f:
    f.write('Resh is a chatbot that answers questions about uploaded documents.')

# Ingest the file
ingest_single_file(sample_path)

# Query the RAG system
answer = get_rag_answer('What is Resh?')
print('RAG Answer:', answer)
