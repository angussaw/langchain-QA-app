"""Python file to serve as the frontend"""
import os
import streamlit as st
import utils as backend
from tempfile import NamedTemporaryFile
import pandas as pd
import json
from st_aggrid import AgGrid, GridOptionsBuilder

def validate_load_PDF(generated_dbs, new_collection_name: str, split_params: dict):
    """_summary_

    Args:
        generated_dbs (_type_): _description_
        new_collection_name (str): _description_
        split_params (dict): _description_

    Returns:
        _type_: _description_
    """
    messages = []

    existing_collection_names = [db["collection name"] for db in generated_dbs]

    if new_collection_name in existing_collection_names:
        messages.append("Collection name already exists")

    if split_params["chunk_size"] <= split_params["chunk_overlap"]:
        messages.append("Chunk size must be greater than chunk overlap")

    return messages


def main():
    """
    """

    hide_default_format = """
        <style>
        .block-container {
                    padding-top: 0.8rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
    
    tooltips = {"chunk_size": "Maximum number of characters that a chunk can contain",
                "chunk_overlap": "Number of characters that should overlap between two adjacent chunks",
                "collection_name": "A short title of the article that acts as a unique identifier for each Chroma vector database of a PDF file. \n \
                                    Refer to https://docs.trychroma.com/usage-guide#creating-inspecting-and-deleting-collections \
                                    for restrictions on Chroma collection names",
                "description": "A short description of the article and its contents",
                "temperature": "A scaling factor that is applied within the final softmax layer that influences the shape \
                                of the probability distribution that the model calculates for the next token. \n \
                                The higher the value, the greater the randomness (vice versa)",
                "select_k_documents": "Number of chunks to be selected as context in the prompt to the LLM"
                }

    # From here down is all the StreamLit UI.
    st.set_page_config(page_title="PDF QA app",
                    layout="wide")
    st.markdown(hide_default_format, unsafe_allow_html=True)

    st.header("Question your PDF!")
    st.caption("Upload a new PDF file with a specified collection name")
    st.caption("Query an existing PDF file by entering your question and OpenAI API key, and specifying the response parameters")
    st.caption("The generated response will provide the best answer, along with k-1 alternative answers for reference")

    existing_chroma_paths = [f.path for f in os.scandir("chroma_db") if f.is_dir()]
    if len(existing_chroma_paths) == 0:
        st.session_state["generated_dbs"] = []

    else:
        if "generated_dbs" not in st.session_state:
            st.session_state["generated_dbs"] = []
            for path in existing_chroma_paths:
                with open(f"{path}/metadata.json") as metadata_json:
                    metadata = json.load(metadata_json)
                st.session_state["generated_dbs"].append(metadata)
                
        generated_db_df = pd.DataFrame(st.session_state["generated_dbs"])
        gb = GridOptionsBuilder.from_dataframe(generated_db_df)
        gb.configure_selection(selection_mode="single", use_checkbox=True)
        gb_grid_options = gb.build()
        st.write("Select a loaded PDF file to query:")
        response = AgGrid(generated_db_df.head(2),gridOptions = gb_grid_options, use_checkbox=True)

    with st.sidebar:
        uploaded_file = st.file_uploader('Upload a PDF file', type='pdf')
        collection_name = st.text_input("Provide article's name", help = tooltips["collection_name"])
        description = st.text_input("Provide article's description", help = tooltips["description"])
        split = st.selectbox('Select splitter', ('CharacterTextSplitter', 'RecursiveCharacterTextSplitter'))
        chunk_size = st.slider('Select chunk size', 100, 1000, 500, help = tooltips["chunk_size"])
        chunk_overlap = st.slider('Select chunk overlap', 50, 500, 250, help = tooltips["chunk_overlap"])

        with st.form('myform', clear_on_submit=True):
            submitted_pdf = st.form_submit_button('Load PDF', disabled=not(uploaded_file))

    if submitted_pdf:
        generated_dbs = st.session_state["generated_dbs"]
        split_params = {"chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "split": split}
        
        messages = validate_load_PDF(generated_dbs = generated_dbs,
                                     new_collection_name = collection_name,
                                     split_params = split_params)

        if len(messages) > 0:
            for message in messages:
                st.write(message)

        else:
            with st.spinner('Loading PDF and creating vector store...'):

                metadata = backend.load_PDF(uploaded_file = uploaded_file,
                                            collection_name = collection_name,
                                            description = description,
                                            split_params = split_params)

                with open(f"chroma_db/{collection_name}/metadata.json", "w") as fp:
                    json.dump(metadata , fp) 
                
                st.session_state.generated_dbs.append(metadata)
                st.write("PDF was loaded successfully")
                st.experimental_rerun()

    if st.session_state["generated_dbs"] != []:
        selected_document = response['selected_rows']

        if selected_document:
            delete_document = st.button("Delete document")

            col1, col2 = st.columns(2, gap="large")

            with col1:
                query_text = st.text_input('Enter your question:')
                openai_api_key = st.text_input('OpenAI API Key', type='password')
                with st.form('myform2', clear_on_submit=True):
                    submitted_query = st.form_submit_button('Generate response', disabled=not(query_text and openai_api_key))

            with col2:
                temperature = st.slider('Select temperature', 0.0, 1.0, 0.7, help = tooltips["temperature"])
                search_type = st.selectbox('Select search type', ('similarity', 'mmr'))
                fetch_k = st.slider('Select k documents', 1, 7, 3, help = tooltips["select_k_documents"])

            if submitted_query and openai_api_key.startswith('sk-'):
                with st.spinner('Retrieving vector store...'):
                    database_search = backend.initialize_vector_database(collection_name = selected_document[0]["collection name"],
                                                                         search_type = search_type,
                                                                         fetch_k = fetch_k)
                    
                    QA_chain, llm = backend.load_retrieval_QA_chain(openai_api_key = openai_api_key,
                                                                    temperature = temperature,
                                                                    retriever=database_search)
                    
                with st.spinner('Initializing agent and generating response...'):
                    agent = backend.initialize_QA_agent(collection_name = selected_document[0]["collection name"],
                                                        description = selected_document[0]["description"],
                                                        chain = QA_chain,
                                                        llm = llm)
                    
                    result = agent({"input":query_text})

                    st.write(result["intermediate_steps"])
                    
            if delete_document:
                collection_name_to_delete = selected_document[0]["collection name"]

                backend.remove_vector_database(collection_name = collection_name_to_delete)

                index_to_remove = [i for i in range(len(st.session_state.generated_dbs)) if st.session_state.generated_dbs[i]["collection name"] == collection_name_to_delete][0]
                st.session_state.generated_dbs.pop(index_to_remove)
                st.experimental_rerun()

if __name__ == "__main__":
    main()
