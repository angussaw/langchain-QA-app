"""Python file to serve as the frontend"""
import os
import streamlit as st
import streamlit_backend as backend
from tempfile import NamedTemporaryFile
import pandas as pd
import json

from st_aggrid import AgGrid, GridOptionsBuilder


# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

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
    response = AgGrid(generated_db_df.head(4),gridOptions = gb_grid_options, use_checkbox=True)

with st.sidebar:
    uploaded_file = st.file_uploader('Upload an article', type='pdf')

    collection_name = st.text_input("Provide article's collection name")

    openai_api_key = st.text_input('OpenAI API Key', type='password')
    # query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)
    split = st.selectbox('Select splitter', ('CharacterTextSplitter', 'RecursiveCharacterTextSplitter'))

    chunk_size = st.slider('Select chunk size', 100, 1000, 500)

    chunk_overlap = st.slider('Select chunk overlap', 50, 500, 250)

    with st.form('myform', clear_on_submit=True):
        submitted_pdf = st.form_submit_button('Load PDF', disabled=not(uploaded_file and openai_api_key))

if submitted_pdf and openai_api_key.startswith('sk-'):

    with st.spinner('Loading PDF and creating vector store...'):
        split_params = {"chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "split": split}
        
        metadata = backend.load_PDF(uploaded_file = uploaded_file,
                                    collection_name = collection_name,
                                    openai_api_key = openai_api_key,
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
        query_text = st.text_input('Enter your question:')
        temperature = st.slider('Select temperature', 0.0, 1.0, 0.7)
        search_type = st.selectbox('Select search type', ('similarity', 'mmr'))
        fetch_k = st.slider('Select k documents', 1, 7, 4)
        with st.form('myform2', clear_on_submit=True):
            submitted_query = st.form_submit_button('Generate response', disabled=not(query_text and openai_api_key))

        if submitted_query and openai_api_key.startswith('sk-'):

            database = backend.initialize_vector_database(collection_name = selected_document[""],
                                                        openai_api_key = openai_api_key)
            
            retrievalQA_chain = backend.load_retrievalQA_chain(db = database,
                                                            openai_api_key = openai_api_key,
                                                            temperature = temperature,
                                                            search_type = search_type,
                                                            fetch_k = fetch_k)

            response = backend.run_retrievalQA_chain(chain = retrievalQA_chain,
                                                    query = query_text)

            st.write(response)

        if delete_document:
            collection_name_to_delete = selected_document[0]["collection name"]

            backend.remove_vector_database(collection_name = collection_name_to_delete)

            index_to_remove = [i for i in range(len(st.session_state.generated_dbs)) if st.session_state.generated_dbs[i]["collection name"] == collection_name_to_delete][0]
            st.session_state.generated_dbs.pop(index_to_remove)
            st.experimental_rerun()




                # chain = backend.load_retrievalQA_chain(db = db,
                #                                        temperature = 0.7,
                #                                        openai_api_key = openai_api_key,
                #                                        search_type = "similarity",
                #                                        fetch_k = 4)
                
                # response = backend.run_retrievalQA_chain(chain = chain,
                #                               query = query_text)
                
                # st.info(response)

               
    # if "generated" not in st.session_state:
#     st.session_state["generated"] = []

# if "past" not in st.session_state:
#     st.session_state["past"] = []


# def get_text():
#     input_text = st.text_input("You: ", "Hello, how are you?", key="input")
#     return input_text


# user_input = get_text()

# if user_input:
#     output = chain.run(input=user_input)

#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(output)

# if st.session_state["generated"]:

#     for i in range(len(st.session_state["generated"]) - 1, -1, -1):
#         message(st.session_state["generated"][i], key=str(i))
#         message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")