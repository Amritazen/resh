import requests, os, time
url = 'http://127.0.0.1:8080/upload'
for i in range(2):
    fname = f'sample{i}.txt'
    with open(fname, 'w') as f:
        f.write(f'Hello {i}')
    with open(fname, 'rb') as f:
        files = {'file': f}
        r = requests.post(url, files=files)
        print('Upload', i, r.status_code, r.text)
    time.sleep(1)
