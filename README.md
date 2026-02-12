# RAG Document Q&A System

A simple Retrieval Augmented Generation (RAG) web application built with Streamlit that allows you to upload documents and ask questions about them.

## Features

- 📄 **Multi-format Support**: Upload PDF, DOCX, and TXT files
- 🔍 **Semantic Search**: Uses sentence transformers for accurate document retrieval
- 🎯 **Source Citations**: Always shows which document chunks were used to answer
- ⚡ **Fast & Lightweight**: Optimized for Streamlit free tier hosting
- 🚫 **Honest Answers**: Only answers when information is in the documents
- 🆓 **Completely Free**: Uses Google Gemini’s generous free tier (no credit card needed)

## Quick Start

### Local Development

1. **Clone or create the project files:**
- `app.py` - Main application code
- `requirements.txt` - Python dependencies
1. **Install dependencies:**
   
   ```bash
   pip install -r requirements.txt
   ```
1. **Get a Google API key (FREE):**
- Visit https://makersuite.google.com/app/apikey
- Sign in with your Google account
- Click “Create API Key”
- No credit card required! Free tier includes 60 requests/minute
1. **Run the app:**
   
   ```bash
   streamlit run app.py
   ```
1. **Use the app:**
- Enter your Google API key in the sidebar
- Upload your documents (PDF, DOCX, or TXT)
- Click “Process Documents”
- Start asking questions!

### Deploy to Streamlit Cloud (Free)

1. **Push to GitHub:**
   
   ```bash
   git init
   git add app.py requirements.txt README.md
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
1. **Deploy on Streamlit Cloud:**
- Go to https://share.streamlit.io/
- Sign in with GitHub
- Click “New app”
- Select your repository
- Set main file path: `app.py`
- Click “Deploy”
1. **Add secrets (optional):**
- In Streamlit Cloud dashboard, go to app settings
- Add secrets in TOML format:
  
  ```toml
  GOOGLE_API_KEY = "your-api-key-here"
  ```
- Modify app.py to read from secrets if preferred

## How It Works

1. **Document Processing**:
- Extracts text from uploaded documents
- Splits text into overlapping chunks (500 words with 50 word overlap)
1. **Embedding Creation**:
- Uses `all-MiniLM-L6-v2` model to create embeddings
- Stores in FAISS vector database for fast similarity search
1. **Query Processing**:
- Converts question to embedding
- Finds top 3 most similar document chunks
- Only uses chunks with >70% similarity
1. **Answer Generation**:
- Sends relevant chunks to Google Gemini (free tier)
- Instructs Gemini to only answer from provided context
- Returns answer with source citations

## Configuration

You can adjust these parameters in `app.py`:

```python
CHUNK_SIZE = 500           # Words per chunk
CHUNK_OVERLAP = 50         # Overlap between chunks
TOP_K_RESULTS = 3          # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7 # Minimum similarity score (0-1)
```

## Limitations (Free Tier)

- **File Size**: Keep documents under 50MB total
- **Memory**: FAISS index stored in memory (limited to ~1-2GB on free tier)
- **API Rate Limits**: Google Gemini free tier allows 60 requests/minute
- **Processing Time**: First upload may take 1-2 minutes for embeddings
- **100% FREE**: Both Streamlit hosting and Google Gemini are completely free!

## Tips for Best Results

1. **Document Quality**: Use well-formatted, text-based documents
1. **Specific Questions**: Ask specific questions rather than broad topics
1. **Document Organization**: Upload related documents together
1. **API Key**: Keep your API key secure, don’t commit it to GitHub

## Troubleshooting

**“Out of memory” error:**

- Reduce `CHUNK_SIZE` or process fewer documents at once
- Consider using smaller embedding models

**Slow processing:**

- Normal for first-time processing
- Subsequent queries are fast

**No relevant results:**

- Try rephrasing your question
- Check if information exists in uploaded documents
- Lower `SIMILARITY_THRESHOLD` if too strict

## Tech Stack

- **Frontend**: Streamlit
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **LLM**: Google Gemini 1.5 Flash (FREE tier)
- **Document Processing**: PyPDF2, python-docx

## License

MIT License - feel free to use and modify!
