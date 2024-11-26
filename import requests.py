import requests

try:
    response = requests.get("http://127.0.0.1:5000/get_pdf_links")
    if response.status_code == 200:
        print("Data received:", response.json())
    else:
        print("Failed with status code:", response.status_code)
except Exception as e:
    print("Error:", e)