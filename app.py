import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def get_pdf_texts(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text   

def get_text_chunks(raw_texts):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_fucntion=len
    )
    chunks = text_splitter.split_text(raw_texts)
    return chunks


def main():
    load_dotenv()
    st.set_page_config(page_title="DocHat", page_icon=':books:')

    st.header("Chat with Multiple PDFs")
    st.text_input("Ask a question abouot your documents")

    with st.sidebar:
        st.subheader("Your documents:")
        pdf_docs = st.file_uploader("Upload your documents:", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Loading..."):
                # get pdfs
                raw_texts = get_pdf_texts(pdf_docs)

                # create text chunks
                chunked_text = get_text_chunks(raw_texts)

                # create vector store 


    

if __name__ == '__main__':
    main()