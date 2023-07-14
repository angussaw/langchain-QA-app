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

from langchain.agents import initialize_agent, Tool, AgentExecutor
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
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separator = " ", length_function = len) 
    
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


def initialize_vector_databases(collection_names: str, fetch_k: int = 4, persist_directory: str = "chroma_db") -> list:
    """_summary_

    Args:
        collection_names (str): _description_
        fetch_k (int, optional): _description_. Defaults to 4.
        persist_directory (str, optional): _description_. Defaults to "chroma_db".

    Returns:
        list: _description_
    """
    vector_databases = []

    embedding_function = HuggingFaceEmbeddings()

    for collection_name in collection_names:
        database_search = Chroma(collection_name = collection_name,
                                persist_directory=f"{persist_directory}/{collection_name}", 
                                embedding_function=embedding_function).as_retriever(search_type="similarity", 
                                                                                    search_kwargs={"k":fetch_k})
        
        vector_databases.append(database_search)
    
    return vector_databases


def load_retrieval_QA_chains(openai_api_key: str, temperature: float, retrievers: list) -> tuple:
    """_summary_

    Args:
        openai_api_key (str): _description_
        temperature (float): _description_
        retrievers (list): _description_

    Returns:
        tuple: _description_
    """
    # "gpt-3.5-turbo" is the default model
    chains = []

    llm = OpenAI(temperature = temperature, openai_api_key = openai_api_key)

    for retriever in retrievers:
        chain = RetrievalQA.from_chain_type(llm = llm,
                                            chain_type = "stuff", # chain_type: specifying how the RetrievalQA should pass the chunks into LLM
                                            retriever = retriever)
        chains.append(chain)

    return chains, llm


def load_PDF(uploaded_file, collection_name: str, description: str, split_params: dict) -> dict:
    """_summary_

    Args:
        uploaded_file (_type_): _description_
        collection_name (str): _description_
        description (str): _description_
        split_params (dict): _description_

    Returns:
        dict: _description_
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


def remove_vector_databases(collection_names: list, persist_directory: str = "chroma_db"):
    """_summary_

    Args:
        collection_names (list): _description_
        persist_directory (str, optional): _description_. Defaults to "chroma_db".
    """
    for collection_name in collection_names:
        shutil.rmtree(f"{persist_directory}/{collection_name}")


def initialize_QA_agents(collection_names: list, descriptions: list, chains: list, llm: OpenAI) -> AgentExecutor:
    """_summary_

    Args:
        collection_names (list): _description_
        descriptions (list): _description_
        chains (list): _description_
        llm (OpenAI): _description_

    Returns:
        AgentExecutor: _description_
    """
    tools = []

    for i in range(len(chains)):

        tool = Tool(name=collection_names[i],
                    func=chains[i].run,
                    description=descriptions[i],
                    return_direct=True)
        
        tools.append(tool)

    agent = initialize_agent(tools,
                             llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True,
                             return_intermediate_steps=True)

    return agent
