"""
## fast_api.py contains the backend logic to process raw input data for inference,
make predictions and generate shap values
"""
from fastapi import FastAPI, UploadFile
import jsonpickle
import os
import pandas as pd
import sys
import uvicorn
import utils as backend
import io
from pathlib import Path
from tempfile import NamedTemporaryFile
import utils
from langchain.document_loaders import PyPDFLoader
import shutil

description = """
api_server API to generate responses from questioning a PDF file.
"""

app = FastAPI(title="api_server",
              description=description,
              version="fastapi:1.0")

openai_api_key = os.getenv("OPENAI_API_KEY")

@app.get("/")
def read_root():
    """Landing Page of API

    Returns:
        JSON: {"content": "FastAPI to generate responses from questioning a PDF file", "version": "<version>",  "model": "<run_id>/<model_uri>"}
    """
    
    return {"content": "FastAPI to generate responses from questioning a PDF file", "version": "1.0"}

@app.post("/load_PDF")
async def load_PDF(collection_name: str, 
                   chunk_size: int,
                   chunk_overlap: int,
                   split: str,
                   filename: str,
                   uploaded_file: UploadFile):
    """_summary_

    Args:
        collection_name (str): _description_
        chunk_size (int): _description_
        chunk_overlap (int): _description_
        split (str): _description_
        filename (str): _description_
        uploaded_file (UploadFile): _description_

    Returns:
        _type_: _description_
    """

    split_params = {"chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "split": split}
    
    path_to_save_file = Path(os.getcwd())
    path_to_save_file.mkdir(parents=True, exist_ok=True)
    file_location = f"{path_to_save_file}/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())

    documents = utils.load_and_split_doc(file_location, **split_params)
    os.unlink(uploaded_file.filename)

    no_of_documents = len(documents)
        
    utils.create_and_persist_vector_database(documents = documents,
                                             collection_name = collection_name)

    metadata = {"filename": filename,
                "collection name": collection_name,
                "split": split_params["split"],
                "chunk_size": split_params["chunk_size"],
                "chunk_overlap": split_params["chunk_overlap"],
                "no_of_documents": no_of_documents}
    
    return metadata

@app.post("/generate_response")
async def generate_response(collection_name: str, 
                            search_type: str,
                            fetch_k: int,
                            temperature: float,
                            query_text: str):
    """_summary_

    Args:
        collection_name (str): _description_
        search_type (str): _description_
        fetch_k (int): _description_
        temperature (float): _description_
        query_text (str): _description_
    """

    database_search = utils.initialize_vector_database(collection_name = collection_name,
                                                       search_type = search_type,
                                                       fetch_k = fetch_k)
    
    QA_chain = utils.load_QA_chain(openai_api_key = openai_api_key,
                                   temperature = temperature)
    
    response = utils.run_QA_chain(chain = QA_chain,
                                  database_search = database_search,
                                  query = query_text)
    
    return response



@app.post("/remove_collection")
async def remove_collection(collection_name: str):
    """_summary_

    Args:
        collection_name (str): _description_
    """
    
    shutil.rmtree(f"chroma_db/{collection_name}")



if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="0.0.0.0", port=8500)