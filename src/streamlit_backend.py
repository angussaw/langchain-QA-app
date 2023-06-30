"""Python file to serve as the backend"""

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationChain, RetrievalQA
from langchain.llms import OpenAI
import yaml
from tempfile import NamedTemporaryFile
import shutil


import logging

# logger = logging.getLogger()

# def setup_logging(
#     logging_config_path="./conf/logging.yaml", default_level=logging.INFO
# ):
#     """Set up configuration for logging utilities.

#     Args:
#         logging_config_path (str, optional): Path to YAML file containing configuration for
#                                              Python logger. Defaults to "./conf/base/logging.yml".
#         default_level (_type_, optional): logging object. Defaults to logging.INFO.
#     """
#     try:
#         with open(logging_config_path, "rt") as file:
#             log_config = yaml.safe_load(file.read())
#             logging.config.dictConfig(log_config)

#     except Exception as error:
#         logging.basicConfig(
#             format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#             level=default_level,
#         )
#         logger.info(error)
#         logger.info("Logging config file is not found. Basic config is being used.")


def load_and_split_doc(filename: str, 
                       chunk_size: int, 
                       chunk_overlap: int, 
                       split: str = "CharacterTextSplitter"):
    """_summary_

    Args:
        filename (str): _description_
        chunk_size (int): _description_
        chunk_overlap (int): _description_

    Returns:
        _type_: _description_
    """
    loader = PyPDFLoader(filename)
    documents = loader.load()

    if split == "CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 
    
    elif split == "RecursiveCharacterTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 

    documents = text_splitter.split_documents(documents)

    return documents


def create_and_persist_vector_database(documents, collection_name: str, openai_api_key: str, persist_directory: str = "chroma_db"):
    """_summary_

    Args:
        documents (_type_): _description_
        collection_name (str): _description_
        persist_directory (str): _description_
    """
    embedding_function = OpenAIEmbeddings(openai_api_key = openai_api_key)

    database = Chroma.from_documents(documents,
                                    embedding = embedding_function,
                                    collection_name = collection_name,
                                    persist_directory = f"{persist_directory}/{collection_name}")
    database.persist()


def initialize_vector_database(collection_name: str, openai_api_key: str, persist_directory: str = "chroma_db"):
    """_summary_

    Args:
        collection_name (str): _description_
        persist_directory (str): _description_

    Returns:
        _type_: _description_
    """
    embedding_function = OpenAIEmbeddings(openai_api_key = openai_api_key)

    database = Chroma(collection_name = collection_name,
                      persist_directory=f"{persist_directory}/{collection_name}", 
                      embedding_function=embedding_function)
    
    return database


def load_retrievalQA_chain(db, openai_api_key: str, temperature: float, search_type: str = "similarity", fetch_k: int = 4):
    """_summary_

    Args:
        db (_type_): _description_
        temperature (float): _description_
        search_type (str, optional): _description_. Defaults to "similarity".
        fetch_k (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """
    llm = OpenAI(temperature = temperature, openai_api_key = openai_api_key)
    chain = RetrievalQA.from_chain_type(chain_type = "stuff",
                                        llm = llm,
                                        retriever = db.as_retriever(search_type = search_type, search_kwargs = {"k": fetch_k}))
    return chain

def run_retrievalQA_chain(chain, query: str):
    """_summary_

    Args:
        chain (_type_): _description_
        query (str): _description_
    """

    result = chain({"query": query})

    return result["result"]


def load_PDF(uploaded_file, collection_name: str, openai_api_key: str, split_params: dict):
    """_summary_

    Args:
        uploaded_file (_type_): _description_
        openai_api_key (str): _description_
        split_params (dict): _description_
    """
    filename = os.path.splitext(uploaded_file.name)[0]

    with NamedTemporaryFile(dir='.', suffix='.pdf', delete=False) as f:
        try:
            f.write(uploaded_file.getbuffer())
            documents = load_and_split_doc(f.name, **split_params)
            f.close()
        finally:
            os.unlink(f.name)

    no_of_documents = len(documents)
        
    create_and_persist_vector_database(documents = documents,
                                       collection_name = collection_name,
                                       openai_api_key = openai_api_key)

    metadata = {"filename": filename,
                "collection name": collection_name,
                "split": split_params["split"],
                "chunk_size": split_params["chunk_size"],
                "chunk_overlap": split_params["chunk_overlap"],
                "no_of_documents": no_of_documents}
    
    return metadata


def remove_vector_database(collection_name: str, persist_directory: str = "chroma_db"):
    """_summary_

    Args:
        collection_name (str): _description_
        persist_directory (str, optional): _description_. Defaults to "chroma_db".
    """
    shutil.rmtree(f"{persist_directory}/{collection_name}")
