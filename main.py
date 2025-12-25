import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


from dotenv import load_dotenv

load_dotenv()

st.title("News Research Tool")
st.sidebar.title("News Article Units")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_URL_clicked = st.sidebar.button("Process URLs")

file_path = 'faiss_store_openai.pkl'

main_placefolder = st.empty()

#creating a llm
llm = OpenAI(temperature = 0.9, max_tokens = 500)

if process_URL_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading.....")
    data = loader.load()

    #split data

    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n','\n','-',','],
        chunk_size = 1000
    )
    main_placefolder.text("Text splitting...")

    docs = text_splitter.split_documents(data)

    #create embedding

    embeddings = OpenAIEmbeddings()
    vectorindex_openai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Creating vectors.....")
    time.sleep(2)

    #saving the FAISS index to a pickle file

    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_openai,f)


query = main_placefolder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result =chain({"Question": query}, return_only_outputs=True)

            #{"answer":"","sources":[]}

            st.header("Answer")
            st.subheader(result["answer"])

            #display sources, if available

            sources = result.get("sorces:")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n") #split by new line

                for sources in sources_list:
                    st.write(sources)