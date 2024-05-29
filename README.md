# PDF-Converse
The PDF Converse tool can be used to upload PDFs and effectively answer all queries. With this tool, users can collaborate and engage in discussions within the context of the document itself. 

# Features
1. Upload PDF files: Users can easily upload their PDF files using the user-friendly interface.
2. Extract text: The application extracts the text content from the uploaded PDF files using PyPDF2 library.
3. Text processing and embeddings: The code utilizes langchain's CharacterTextSplitter to segment the text into manageable chunks. Additionally, it employs OllamaEmbeddings to generate semantic embeddings, enhancing the understanding of the text's context and enabling more effective analysis and retrieval.
4. Building the knowledge base: The text chunks and embeddings are stored in Chroma, leveraging FAISS for efficient similarity search, enhancing retrieval accuracy. User interaction and question answering: Users can input questions, and the application utilizes a question answering chain, coupled with the knowledge base, to deliver accurate responses, enhancing user experience.

# Installation
1. Clone the repository:
git clone https://github.com/Bhavd33p/PDF-Converse.git
Install the required dependencies:
# Note There are two python files 
~ pdfconverse.py {This is best model for this PS and application will run on this}
~ chat.py {This is CLI based model}
2. pip install -r requirements.txt
3. After Installing all dependencies you need to install ollama from https://ollama.com/ for Command Line 
4. Then use these command to fetch the model :
~ ollama pull nomic-embed-text
~ ollama run mistral

# Usage 
Run the Application by 
1. streamlit run pdfconverse.py 
if Axios error 403 occurs upon uploading pdf use this : streamlit run pdfconverse.py --server.enableXsrfProtection false
2. Upload your PDF file using upload pdf section in sidebar.
3. Ask a query about the PDF content in the chat options.
4. The application will display the answer based on the uploaded PDF file.

