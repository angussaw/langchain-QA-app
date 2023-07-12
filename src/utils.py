"""Python file to serve as the backend"""

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, OpenAI
from langchain.chains import ConversationChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from tempfile import NamedTemporaryFile
import shutil

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


def create_and_persist_vector_database(documents, collection_name: str, persist_directory: str = "chroma_db"):
    """_summary_

    Args:
        documents (_type_): _description_
        collection_name (str): _description_
        persist_directory (str, optional): _description_. Defaults to "chroma_db".
    """

    embedding_function = HuggingFaceEmbeddings()

    database = Chroma.from_documents(documents,
                                        embedding = embedding_function,
                                        collection_name = collection_name,
                                        persist_directory = f"{persist_directory}/{collection_name}")
    database.persist()
        


def initialize_vector_database(collection_name: str, search_type: str = "similarity", fetch_k: int = 4, persist_directory: str = "chroma_db"):
    """_summary_

    Args:
        collection_name (str): _description_
        search_type (str, optional): _description_. Defaults to "similarity".
        fetch_k (int, optional): _description_. Defaults to 4.
        persist_directory (str, optional): _description_. Defaults to "chroma_db".

    Returns:
        _type_: _description_
    """

    embedding_function = HuggingFaceEmbeddings()

    database_search = Chroma(collection_name = collection_name,
                             persist_directory=f"{persist_directory}/{collection_name}", 
                             embedding_function=embedding_function).as_retriever(search_type=search_type, search_kwargs={"k":fetch_k})
    
    return database_search


def load_QA_chain(openai_api_key: str, temperature: float):
    """_summary_

    Args:
        openai_api_key (str): _description_
        temperature (float): _description_

    Returns:
        _type_: _description_
    """
    # "gpt-3.5-turbo" is the default model

    llm = OpenAI(temperature = temperature, openai_api_key = openai_api_key)
    chain = load_qa_chain(llm=llm,
                          chain_type="map_rerank", # chain_type: specifying how the RetrievalQA should pass the chunks into LLM
                          return_intermediate_steps=True)
    return chain

def load_retrieval_QA_chain(openai_api_key: str, temperature: float, retriever):
    """_summary_

    Args:
        openai_api_key (str): _description_
        temperature (float): _description_
        retriever (_type_): _description_

    Returns:
        _type_: _description_
    """
    # "gpt-3.5-turbo" is the default model

    llm = OpenAI(temperature = temperature, openai_api_key = openai_api_key)
    chain = RetrievalQA.from_chain_type(llm = llm,
                                        chain_type = "stuff", # chain_type: specifying how the RetrievalQA should pass the chunks into LLM
                                        retriever = retriever)

    return chain, llm

def run_QA_chain(chain, database_search, query: str):
    """_summary_

    Args:
        chain (_type_): _description_
        database_search (_type_): _description_
        query (str): _description_

    Returns:
        _type_: _description_
    """

    docs = database_search.get_relevant_documents(query)
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

    return result


def load_PDF(uploaded_file, collection_name: str, description: str, split_params: dict):
    """_summary_

    Args:
        uploaded_file (_type_): _description_
        collection_name (str): _description_
        description (str): _description_
        split_params (dict): _description_

    Returns:
        _type_: _description_
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
                                       collection_name = collection_name)

    metadata = {"filename": filename,
                "collection name": collection_name,
                "description": description,
                "no_of_documents": no_of_documents}
    
    return metadata


def remove_vector_database(collection_name: str, persist_directory: str = "chroma_db"):
    """_summary_

    Args:
        collection_name (str): _description_
        persist_directory (str, optional): _description_. Defaults to "chroma_db".
    """
    shutil.rmtree(f"{persist_directory}/{collection_name}")


def initialize_QA_agent(collection_name: str, description: str, chain, llm):
    """_summary_

    Args:
        collection_name (str): _description_
        description (str): _description_
        chain (_type_): _description_
        llm (_type_): _description_

    Returns:
        _type_: _description_
    """
    tools = [Tool(name=collection_name,
                  func=chain.run,
                  description=description,
                  return_direct=True)]

    agent = initialize_agent(tools,
                             llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True,
                             return_intermediate_steps=True)

    return agent