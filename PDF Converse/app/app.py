
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin  
from PyPDF2 import PdfReader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
import logging

app = Flask(__name__)
CORS(app)  

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

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
    vector_database = Chroma.from_texts(
        texts=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="document-collection"
    )
    model_name = "mistral"
    language_model = ChatOllama(model=model_name)
    return vector_database, language_model

@app.route('/process_pdfs', methods=['POST'])
@cross_origin(origin='chrome-extension://kfaiomnogeopahmlldhpmhehncagpjen', headers=['Content-Type'])
def process_pdfs():
    uploaded_files = request.files.getlist("pdf_files")
    query = request.form.get("query")

    if not uploaded_files or not query:
        return jsonify({"error": "Please provide PDF files and a query."}), 400

    saved_files = []
    for file in uploaded_files:
        filepath = os.path.join("/tmp", file.filename)
        file.save(filepath)
        saved_files.append(filepath)

    try:
        text = extract_text_from_pdfs(saved_files)
        vector_db, llm = process_text_chunks(text)

        # Define a query prompt template
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant.Your task is to generate three
            different versions of the given user question to retrieve relevant documents from
            a vector database.By generating multiple perspectives on the user query question,your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search.Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), 
            llm,
            prompt=QUERY_PROMPT
        )

        template = f"""Answer the question based ONLY on the following context:
        {{context}}
        Question: {query}"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke(query)

        vector_db.delete_collection()

        logging.info(f"Query processed successfully. Query: {query}, Answer: {answer}")

        return jsonify({"answer": answer})

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    finally:
        for filepath in saved_files:
            os.remove(filepath)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
