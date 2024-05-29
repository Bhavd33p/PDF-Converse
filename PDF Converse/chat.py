import os
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sklearn.metrics.pairwise import pairwise_distances
from transformers import BertTokenizer, BertForMaskedLM, GPT2LMHeadModel, GPT2Tokenizer
import torch
import nltk
import ssl

import logging
logging.basicConfig(level=logging.ERROR)
# for nltk ssl certification if it shows error 
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def load_glove_vectors(filepath):
    glove_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            glove_dict[word] = vector
    return glove_dict

def preprocess_text(text):
    text = text.lower()
    tokens = text.split()
    if not tokens:
        return []  
    
    tokens = [token for token in tokens if not token.isdigit()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def get_embedding(tokens, glove_dict, vector_size=300):
    embeddings = []
    for token in tokens:
        if token in glove_dict:
            embeddings.append(glove_dict[token])
        else:
            embeddings.append(np.zeros(vector_size))
    return np.mean(embeddings, axis=0)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def generate_context(input_text, knowledge, model, tokenizer, max_length=1024):
    input_text_with_knowledge = input_text + "\n" + knowledge
    inputs = tokenizer.encode_plus(input_text_with_knowledge, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def process_pdfs(pdf_docs, glove_vectors):
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    print("Number of text chunks extracted:", len(text_chunks))
    
    text_list = text_chunks
    embeddings = np.array([get_embedding(preprocess_text(text), glove_vectors) for text in text_chunks])
    return embeddings, text_list

def main():
    pdf_folder = input("Enter the folder path containing PDFs: ")
    pdf_docs = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith(".pdf")]

    glove_file_path = 'glove.42B.300d.txt'
    glove_vectors = load_glove_vectors(glove_file_path)
    
    embeddings, text_list = process_pdfs(pdf_docs, glove_vectors)
    print("PDF processing complete.")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    query = input("Ask a Question about your Documents: ")
    query_embedding = get_embedding(preprocess_text(query), glove_vectors)
    query_embedding = query_embedding.reshape(1, -1)

    similarities = pairwise_distances(embeddings, query_embedding, metric='cosine')
    indices = np.argsort(similarities[:, 0])[:1]
    distances = similarities[indices]

    print("Best matched document:")
    print(f"Text: {text_list[indices[0]]}, Similarity: {1 - distances[0][0]}")
    
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    input_text = text_list[indices[0]] 
    additional_knowledge = "Additional knowledge to incorporate into the context."
    context = generate_context(input_text, additional_knowledge, gpt2_model, gpt2_tokenizer)
    print(f"Context: {context[:20000]}")  

if __name__ == '__main__':
    main()
