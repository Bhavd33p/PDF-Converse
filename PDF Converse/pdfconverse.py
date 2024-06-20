import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import time

def extract_text_from_pdfs(uploaded_pdfs):
    combined_text = ""
    for pdf in uploaded_pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            combined_text += page.extract_text()
    return combined_text

def process_text_chunks(text_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text_data)
    vector_store = Chroma.from_texts(
        texts=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="document-collection"
    )
    model_name = "mistral"
    language_model = ChatOllama(model=model_name)
    return vector_store, language_model

def main():
    st.set_page_config(page_title="PDF Converse 📚", page_icon="📚")
    st.markdown("""
        <style>
        .main-title {
            text-align: left;
            font-size: 2.5em;
            color: #4CAF50;
            font-weight: bold;
            padding-bottom: 20px;
        }
        .sidebar-title {
            font-size: 1.5em;
            color: #4CAF50;
        }
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            color: #888;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">PDF Converse 📚</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="sidebar-title">Upload PDF</div>', unsafe_allow_html=True)
    pdf_files = st.sidebar.file_uploader("Choose PDF Files", type=["pdf"], accept_multiple_files=True)
    
    st.sidebar.markdown('<div class="sidebar-title">Chat Options</div>', unsafe_allow_html=True)
    query = st.sidebar.text_input("Enter your query:", "")
    st.markdown("""
        <div class="footer">
            Made with ❤️ by Bhavdeep
        </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("Process"):
        if pdf_files:
            with st.spinner("Processing PDF..."):
                progress_bar = st.progress(0)
                time.sleep(1)

                combined_text = extract_text_from_pdfs(pdf_files)
                progress_bar.progress(33)
                time.sleep(1)
                
                vector_store, language_model = process_text_chunks(combined_text)
                progress_bar.progress(66)
                time.sleep(1)

                QUERY_PROMPT = PromptTemplate(
                    input_variables=["question"],
                    template="""As your AI language assistant, I'll create three distinct variations of your query to find relevant documents from a vector database. By offering diverse perspectives, I aim to help you overcome limitations of distance-based similarity search methods. Please list these alternative questions separately.
                    Original query: {question}""",
                )
                retriever = MultiQueryRetriever.from_llm(
                    vector_store.as_retriever(), 
                    language_model,
                    prompt=QUERY_PROMPT
                )

                ANSWER_PROMPT_TEMPLATE = f"""Based on the following context, answer the question as accurately as possible:
                Context: {{context}}
                Question: {query}"""
                prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT_TEMPLATE)
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | language_model
                    | StrOutputParser()
                )

                progress_bar.progress(100)
                time.sleep(1)
                st.write("Answer:", chain.invoke(query))

                vector_store.delete_collection()
        else:
            st.warning("Please upload PDF files.")

if __name__ == "__main__":
    main()
