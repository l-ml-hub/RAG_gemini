import streamlit as st
import os
from pathlib import Path
import pickle
from typing import List, Dict, Tuple
import re

# Document processing

import PyPDF2
from docx import Document as DocxDocument

# Embeddings and vector store

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# LLM

import google.generativeai as genai

# Configuration

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3
SIMILARITY_THRESHOLD = 0.7

# Initialize session state

if 'vector_store' not in st.session_state:
st.session_state.vector_store = None
if 'chunks' not in st.session_state:
st.session_state.chunks = []
if 'doc_metadata' not in st.session_state:
st.session_state.doc_metadata = []
if 'embeddings_model' not in st.session_state:
st.session_state.embeddings_model = None

@st.cache_resource
def load_embeddings_model():
"""Load the sentence transformer model for embeddings."""
return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file) -> str:
"""Extract text from PDF file."""
pdf_reader = PyPDF2.PdfReader(file)
text = ""
for page in pdf_reader.pages:
text += page.extract_text() + "\n"
return text

def extract_text_from_docx(file) -> str:
"""Extract text from DOCX file."""
doc = DocxDocument(file)
text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
return text

def extract_text_from_txt(file) -> str:
"""Extract text from TXT file."""
return file.read().decode('utf-8')

def process_document(file) -> Tuple[str, str]:
"""Process uploaded document and extract text."""
file_extension = Path(file.name).suffix.lower()

```
if file_extension == '.pdf':
    text = extract_text_from_pdf(file)
elif file_extension == '.docx':
    text = extract_text_from_docx(file)
elif file_extension == '.txt':
    text = extract_text_from_txt(file)
else:
    raise ValueError(f"Unsupported file type: {file_extension}")

return text, file.name
```

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
"""Split text into overlapping chunks."""
words = text.split()
chunks = []

```
for i in range(0, len(words), chunk_size - overlap):
    chunk = ' '.join(words[i:i + chunk_size])
    if chunk:
        chunks.append(chunk)

return chunks
```

def create_vector_store(chunks: List[str], metadata: List[Dict], model):
"""Create FAISS vector store from text chunks."""
embeddings = model.encode(chunks, show_progress_bar=True)

```
# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
index.add(embeddings.astype('float32'))

return index, embeddings
```

def search_similar_chunks(query: str, index, chunks: List[str], metadata: List[Dict],
model, top_k: int = TOP_K_RESULTS) -> List[Dict]:
"""Search for similar chunks using vector similarity."""
query_embedding = model.encode([query])
faiss.normalize_L2(query_embedding)

```
scores, indices = index.search(query_embedding.astype('float32'), top_k)

results = []
for score, idx in zip(scores[0], indices[0]):
    if score >= SIMILARITY_THRESHOLD:
        results.append({
            'chunk': chunks[idx],
            'metadata': metadata[idx],
            'score': float(score)
        })

return results
```

def generate_answer(query: str, context_chunks: List[Dict], api_key: str) -> str:
"""Generate answer using Google Gemini API."""
if not context_chunks:
return “I couldn’t find relevant information in the uploaded documents to answer your question. Please check the documents or upload more relevant materials.”

```
# Build context from retrieved chunks
context = "\n\n".join([
    f"[Source: {chunk['metadata']['source']}, Chunk {chunk['metadata']['chunk_id']}]\n{chunk['chunk']}"
    for chunk in context_chunks
])

prompt = f"""Based on the following context from the documents, please answer the question. 
```

IMPORTANT INSTRUCTIONS:

- Only use information that is explicitly stated in the context below
- If the answer is not in the context, say “I don’t have enough information in the documents to answer this question” and recommend which document or section the user should read
- Always cite the source document when providing information
- Be concise and accurate

Context:
{context}

Question: {query}

Answer:”””

```
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text
except Exception as e:
    return f"Error generating answer: {str(e)}"
```

def main():
st.set_page_config(
page_title=“RAG Document Q&A”,
page_icon=“📚”,
layout=“wide”
)

```
st.title("📚 RAG Document Q&A System")
st.markdown("Upload your documents and ask questions. The system will only answer based on the uploaded content.")

# Sidebar for configuration and document upload
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Try to get API key from secrets first, then from user input
    api_key = None
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY", None)
    except:
        pass
    
    if not api_key:
        api_key = st.text_input(
            "Google API Key (Free)",
            type="password",
            help="Enter your Google API key. Get one FREE at https://makersuite.google.com/app/apikey"
        )
    else:
        st.success("✅ API Key loaded from secrets")
    
    st.markdown("---")
    st.header("📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            # Load embeddings model
            if st.session_state.embeddings_model is None:
                st.session_state.embeddings_model = load_embeddings_model()
            
            all_chunks = []
            all_metadata = []
            
            for file in uploaded_files:
                try:
                    text, filename = process_document(file)
                    chunks = chunk_text(text)
                    
                    # Create metadata for each chunk
                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        all_metadata.append({
                            'source': filename,
                            'chunk_id': i,
                            'total_chunks': len(chunks)
                        })
                    
                    st.success(f"✅ Processed {filename}: {len(chunks)} chunks")
                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {str(e)}")
            
            if all_chunks:
                # Create vector store
                with st.spinner("Creating vector store..."):
                    index, embeddings = create_vector_store(
                        all_chunks,
                        all_metadata,
                        st.session_state.embeddings_model
                    )
                    st.session_state.vector_store = index
                    st.session_state.chunks = all_chunks
                    st.session_state.doc_metadata = all_metadata
                
                st.success(f"🎉 Successfully processed {len(uploaded_files)} documents with {len(all_chunks)} total chunks!")
    
    # Display current status
    if st.session_state.vector_store is not None:
        st.markdown("---")
        st.header("📊 Status")
        st.metric("Documents Loaded", len(set([m['source'] for m in st.session_state.doc_metadata])))
        st.metric("Total Chunks", len(st.session_state.chunks))

# Main chat interface
st.header("💬 Ask Questions")

if st.session_state.vector_store is None:
    st.info("👈 Please upload and process documents in the sidebar to get started.")
else:
    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📎 View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"""
                        **Source:** {source['metadata']['source']}  
                        **Chunk:** {source['metadata']['chunk_id'] + 1}/{source['metadata']['total_chunks']}  
                        **Similarity:** {source['score']:.2%}
                        
                        ---
                        {source['chunk'][:300]}...
                        """)
    
    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        if not api_key:
            st.error("⚠️ Please enter your Google API key in the sidebar.")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    # Search for relevant chunks
                    results = search_similar_chunks(
                        query,
                        st.session_state.vector_store,
                        st.session_state.chunks,
                        st.session_state.doc_metadata,
                        st.session_state.embeddings_model
                    )
                    
                    # Generate answer
                    answer = generate_answer(query, results, api_key)
                    st.markdown(answer)
                    
                    # Show sources
                    if results:
                        with st.expander("📎 View Sources"):
                            for result in results:
                                st.markdown(f"""
                                **Source:** {result['metadata']['source']}  
                                **Chunk:** {result['metadata']['chunk_id'] + 1}/{result['metadata']['total_chunks']}  
                                **Similarity:** {result['score']:.2%}
                                
                                ---
                                {result['chunk'][:300]}...
                                """)
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": results
                    })
```

if **name** == “**main**”:
main()
