import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import openai
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from htmlTemplates import css, bot_template, user_template
import os
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

# def delete_documents_contents():
#     folder_path = "documents"
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         try:
#             if os.path.isfile(file_path):
#                 os.unlink(file_path)
#         except Exception as e:
#             print(f"Failed to delete {file_path}. Reason: {e}")

# atexit.register(delete_documents_contents)

def file_loader(directory):
    read_doc = PyPDFDirectoryLoader(directory)
    documents = read_doc.load()
    return documents

def is_documents_folder_empty():
    return len(os.listdir("documents")) == 0

def chunk_data(doc):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(doc)
    return docs

def get_vectorstore(text_chunks):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'], model="text-embedding-3-small")

    # Create or get Pinecone index
    index_name = 'dochat'
    # if index_name not in pc.list_indexes().names():
    #     pc.create_index(
    #         name=index_name,
    #         dimension=embeddings.embedding_dimension(),
    #         metric='cosine'
    #     )

    docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)
    # query = "What Skills make Rajat a good fit for a Software Engineering Internship?"
    # docs = docsearch.similarity_search(query)
    # st.write(docs[0].page_content)
    return docsearch

def create_conversation_chain(vectorestore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # Use RetrievalChain from retrieval_transformers
    llm = ChatOpenAI()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    # st.write(response)
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0 :
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else :
             st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="DocHat", page_icon=':books:')
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None    

    st.header("Chat with Multiple PDFs")
    user_question = st.text_input("Ask a question about your documents")
    if user_question:
        handle_input(user_question)

    # st.write(user_template.replace("{{MSG}}", "Hello Robot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    

    with st.sidebar:
        st.subheader("Your documents:")
        pdf_docs = st.file_uploader("Upload your documents:", accept_multiple_files=True)
        # st.write(pdf_docs)
        if st.button("Upload"):
            with st.spinner("Loading..."):
                for pdf_file in pdf_docs:
                        file_name = pdf_file.name
                        if not os.path.exists(os.path.join("documents", file_name)):
                            with open(os.path.join("documents", file_name), "wb") as f:
                                f.write(pdf_file.read())
                                # st.write('Done')

                docs = file_loader('documents/')
                # st.write(docs)
                
                chunked_text = chunk_data(docs)      
                # st.write(chunked_text)

                # Create vector store 
                vectorestore = get_vectorstore(chunked_text)

                # Create conversation chain
                st.session_state.conversation = create_conversation_chain(vectorestore)
                

                

if __name__ == '__main__':
    main()
    # embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
    # print(embeddings.embed_query('How are youu?'))