# 📄 PDF Guided RAG System

### High-Performance Retrieval-Augmented Generation for Large Documents

------------------------------------------------------------------------

## 🚀 Project Overview

This project is a **production-grade Retrieval-Augmented Generation
(RAG) system** that allows users to upload large PDF documents
(including 3000--5000+ page books) and interactively ask questions about
their contents.

The system combines:

-   ⚡ Groq LLM (`openai/gpt-oss-120b`)
-   🧠 Local GPU embeddings (HuggingFace)
-   🔍 FAISS vector database
-   🧩 Hybrid chunking strategy
-   🚀 Parallel batch embedding
-   💾 Smart caching
-   🧮 LaTeX math rendering
-   🗑 Automatic temporary file cleanup

Designed to handle **large-scale documents efficiently** while
maintaining high answer quality.

------------------------------------------------------------------------

# 🎯 Why This Project?

Large PDF documents present challenges:

-   Slow embedding times
-   Token overflow issues
-   Redundant context retrieval
-   High runtime cost
-   Poor mathematical formatting
-   Storage management issues

This system solves these problems by implementing:

✔ Hybrid semantic chunking\
✔ GPU-accelerated embeddings\
✔ Concurrent batch processing\
✔ MMR-based smart retrieval\
✔ Context token limiting\
✔ Ephemeral caching system\
✔ Automatic file cleanup

------------------------------------------------------------------------

# 🏗 System Architecture

PDF Upload\
↓\
PDF → Markdown Conversion\
↓\
Markdown Header Split\
↓\
Semantic Chunking\
↓\
Recursive Chunking\
↓\
GPU Batch Embedding (Parallelized)\
↓\
FAISS Vector Store\
↓\
MMR Retriever\
↓\
Groq LLM\
↓\
Formatted Response (LaTeX supported)

------------------------------------------------------------------------

# 🛠 Tech Stack

-   Python 3.10+
-   FastAPI
-   Streamlit
-   LangChain
-   Groq API
-   HuggingFace Transformers
-   FAISS
-   PyMuPDF
-   CUDA (optional but recommended)

------------------------------------------------------------------------

# 🔑 Setting Up Groq API

Follow these steps to configure Groq:

### 1️⃣ Create a Groq Account

Go to: https://console.groq.com/

Sign up or log in.

------------------------------------------------------------------------

### 2️⃣ Generate an API Key

-   Navigate to **API Keys**
-   Click **Create API Key**
-   Copy the generated key

------------------------------------------------------------------------

### 3️⃣ Add API Key to Project

Create a file named:

    config.py

Add:

``` python
GROQ_API_KEY = "your_groq_api_key_here"
```

Alternatively, you can set it as an environment variable:

Windows:

    set GROQ_API_KEY=your_key_here

Mac/Linux:

    export GROQ_API_KEY=your_key_here

------------------------------------------------------------------------

# 📦 Installation & Setup

## 1️⃣ Clone the Repository

    git clone https://github.com/yourusername/pdf-guided-rag.git
    cd pdf-guided-rag

------------------------------------------------------------------------

## 2️⃣ Create Virtual Environment

    python -m venv rag_env

Activate:

Windows:

    rag_env\Scripts\activate

Mac/Linux:

    source rag_env/bin/activate

------------------------------------------------------------------------

## 3️⃣ Install Dependencies

    pip install -r requirements.txt

------------------------------------------------------------------------

# ▶ Running the Application

## Start Backend (FastAPI)

    cd backend
    uvicorn main:app --port 8001

------------------------------------------------------------------------

## Start Frontend (Streamlit)

    cd frontend
    streamlit run app.py

Open in browser:

http://localhost:8501

------------------------------------------------------------------------

# ⚙ Performance Optimizations

-   Batch Embedding
-   Parallel Processing (ThreadPoolExecutor)
-   GPU Acceleration
-   MMR Retrieval
-   Token Limiting
-   Smart Caching
-   Temporary File Cleanup

------------------------------------------------------------------------

# 📊 Performance Benchmarks

  Document Size   First Load   Cached Load
  --------------- ------------ -------------
  500 pages       \~30 sec     \<1 sec
  3000 pages      2--4 min     \<2 sec
  5000 pages      \~5 min      \<3 sec

------------------------------------------------------------------------

# 📚 Learning Outcomes

-   Advanced RAG architecture
-   Token management strategies
-   Retrieval optimization
-   GPU embedding acceleration
-   Production-grade backend design
-   Resource lifecycle management
-   Parallel processing in Python

------------------------------------------------------------------------

# 📄 License

MIT License

------------------------------------------------------------------------

⭐ If you found this useful, consider starring the repository.
