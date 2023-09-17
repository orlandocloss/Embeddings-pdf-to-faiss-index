import os
import numpy as np
import faiss
import PyPDF2
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Set the model to evaluation mode

class PDFProcessorWithHuggingFace:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.content = []
        self.chunk_size_chars=1200

    def read_pdf_content(self):
        with open(self.pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for index,page_num in enumerate(range(len(reader.pages))):
                self.content.append(reader.pages[page_num].extract_text())

    def split_content_into_chunks(self):
        str_content=''.join(self.content)
        chunks=[]
        for i in range(0, len(str_content), self.chunk_size_chars):
            chunk=str_content[i:(i+self.chunk_size_chars)]
            chunks.append(chunk)
        # print(chunks)
        return chunks

    def get_embeddings_from_huggingface(self, text):
        # Tokenize input text
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract the last hidden states (embeddings)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings

    def embed_chunks(self, chunks):
        return [self.get_embeddings_from_huggingface(chunk) for chunk in chunks]

    def create_faiss_index(self, embeddings):
        embeddings_array = np.array(embeddings).reshape(len(embeddings), -1)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        return index

    def process_pdf(self):
        self.read_pdf_content()
        chunks = self.split_content_into_chunks()
        # Save the chunks to a separate file for quick retrieval later
        with open("/path/to/chunks.txt", 'w') as f:
            # print(chunks)
            for chunk in chunks:
                f.write(chunk + "\n====\n")
        embeddings = self.embed_chunks(chunks)
        index = self.create_faiss_index(embeddings)
        return index

pdf_path = "/path/to/input"  # Replace with your PDF path
processor = PDFProcessorWithHuggingFace(pdf_path)
faiss_index = processor.process_pdf()

# To save the Faiss index:
faiss.write_index(faiss_index, "/path/to/output")
