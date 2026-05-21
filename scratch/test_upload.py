import requests, os

url = 'http://localhost:8080/upload'
file_path = os.path.abspath('sample_upload.txt')
with open(file_path, 'rb') as f:
    files = {'file': f}
    resp = requests.post(url, files=files)
    print('Status:', resp.status_code)
    print('Response:', resp.json())
