import os
import requests
from streamlit.runtime.uploaded_file_manager import UploadedFile

# the SERVER_URL matches the name of the service
SERVER_URL = "http://127.0.0.1:8000"


def get_fastapi_status(server_url: str = SERVER_URL):
    """Access FastAPI /docs endpoint to check if server is running"""
    try:
        response = requests.get(f"{server_url}/docs")
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        return False


def set_models(embed_model: str, ia_model: str):
    url = f'{SERVER_URL}/models?embed_mode={embed_model}&llm_model={ia_model}'
    response = requests.get(url)
    return response


def temporal_save(files: list[UploadedFile]):
    for i in range(len(files)):
        bytes_data = files[i].read()  # read the content of the file in binary
        with open(os.path.join("./upload_docs", files[i].name), "wb") as f:
            f.write(bytes_data)       # write this content elsewhere


def temporal_delete():
    for fitxer in os.listdir('./upload_docs'):
        os.remove(os.path.join('./upload_docs', fitxer))


def post_store_pdfs(pdf_files: list[UploadedFile], server_url: str = SERVER_URL):
    """Send POST request to FastAPI /documents endpoint"""
    temporal_save(pdf_files)
    response = requests.post(f"{SERVER_URL}/documents")
    temporal_delete()
    return response


def get_all_documents_file_name():
    """Send GET request to FastAPI /documents_stored endpoint"""
    response = requests.get(f"{SERVER_URL}/documents_stored")
    return response


def get_rag_answer(rag_query: str, server_url: str = SERVER_URL):
    """Send GET request to FastAPI /rag_answer endpoint"""
    url = f'{server_url}/pregunta?rag_query={rag_query}'
    response = requests.get(url)
    return response
