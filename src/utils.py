"""Python file to serve as the backend"""

import os
from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import  RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentExecutor, AgentType
from langchain.memory import ConversationBufferMemory

from src.prompts import qa_template
from tempfile import NamedTemporaryFile
import shutil
import yaml

# Import config vars
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

def load_and_split_doc(filename: str, 
                       chunk_size: int, 
                       chunk_overlap: int):
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

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                            separator = " ", length_function = len) 

    documents = text_splitter.split_documents(documents)

    return documents


def create_and_persist_vector_database(documents, collection_name: str):
    """_summary_

    Args:
        documents (_type_): _description_
        collection_name (str): _description_
    """

    embedding_function = HuggingFaceEmbeddings()

    database = Chroma.from_documents(documents,
                                        embedding = embedding_function,
                                        collection_name = collection_name,
                                        persist_directory = f"{config['DB_CHROMA_PATH']}/{collection_name}")
    database.persist()


def initialize_vector_databases(collection_names: str, fetch_k: int = 4) -> list:
    """_summary_

    Args:
        collection_names (str): _description_
        fetch_k (int, optional): _description_. Defaults to 4.

    Returns:
        list: _description_
    """
    vector_databases = []

    embedding_function = HuggingFaceEmbeddings() # sentence transformer (vector of 768 dimension)

    for collection_name in collection_names:
        database_search = Chroma(collection_name = collection_name,
                                persist_directory=f"{config['DB_CHROMA_PATH']}/{collection_name}", 
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
    
    chains = []

    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])

    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = temperature, openai_api_key = openai_api_key) # gpt-3.5-turbo is the default chat model

    for retriever in retrievers:
        chain = RetrievalQA.from_chain_type(llm = llm, # gpt-3.5-turbo
                                            chain_type = "stuff", # chain_type: specifying how the RetrievalQA should pass the chunks into LLM
                                            retriever = retriever,
                                            chain_type_kwargs={'prompt': prompt})

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

    description = description + " Input should be a fully formed question."
    metadata = {"filename": filename,
                "collection name": collection_name,
                "description": description,
                "no_of_documents": no_of_documents}
    
    return metadata


def remove_vector_databases(collection_names: list):
    """_summary_

    Args:
        collection_names (list): _description_
    """
    for collection_name in collection_names:
        shutil.rmtree(f"{config['DB_CHROMA_PATH']}/{collection_name}")


def initialize_conversational_react_agent(collection_names: list, descriptions: list, chains: list, llm: ChatOpenAI) -> AgentExecutor:
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
    # PREFIX = """You are an expert career coach, good at reviewing resumes."""

    for i in range(len(chains)):

        tool = Tool(name=collection_names[i],
                    func=chains[i].run,
                    description=descriptions[i],
                    return_direct=True)
        
        tools.append(tool)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(tools, llm, 
                             agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                             verbose=True, memory=memory, handle_parsing_errors=True)
                            #  agent_kwargs={'prefix':PREFIX})
        # 'format_instructions':FORMAT_INSTRUCTIONS,
        # 'suffix':SUFFIX

    return agent


