import os
import fitz
import pickle
import hashlib
import tempfile
import atexit
import shutil
from concurrent.futures import ThreadPoolExecutor

from config import GROQ_API_KEY

# LangChain core
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Groq LLM
from langchain_groq import ChatGroq

# Embeddings (local)
from langchain_huggingface import HuggingFaceEmbeddings

# Vector store + loader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

# Set API key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Cache directory cleanup on exit
CACHE_DIR = tempfile.mkdtemp(prefix="rag_cache_")

# Cleanup function to remove cache directory on exit
def cleanup():

    shutil.rmtree(CACHE_DIR, ignore_errors=True)

atexit.register(cleanup)

#Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda"}
)

# GROQ model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    max_tokens=2048
)

# Prompt
prompt = PromptTemplate.from_template(
"""
You are an expert document analyst.

Your job is to answer questions using ONLY the provided document context.

INSTRUCTIONS:

• Provide a COMPLETE answer
• Provide proper explanation as asked (Detailed,brief, step-by-step, etc.)
• Format mathematical formulas using LaTeX between $$ symbols
• Provide STRUCTURED output when possible
• Use bullet points when appropriate
• Use headings if helpful
• Include all relevant information from the document

STRICT RULE:

If answer is not present in context, say:
"Answer not found in document"

---------------------

DOCUMENT CONTEXT:
{context}

---------------------

QUESTION:
{question}

---------------------

ANSWER:
"""
)

# Cache path generator
def get_cache_path(pdf_path):

    file_hash = hashlib.md5(
        open(pdf_path, "rb").read()
    ).hexdigest()

    return os.path.join(CACHE_DIR, f"{file_hash}.pkl")

# Batch embedding function
def embed_batches(docs, batch_size=32):

    texts = [
        doc.page_content
        for doc in docs
    ]

    batches = [
        texts[i:i + batch_size]
        for i in range(0, len(texts), batch_size)
    ]

    embeddings = []

    with ThreadPoolExecutor(max_workers=4) as executor:

        results = executor.map(
            embedding.embed_documents,
            batches
        )

        for batch in results:

            embeddings.extend(batch)

    return embeddings

#PDF to Markdown converter
def pdf_to_markdown(pdf_path):

    doc = fitz.open(pdf_path)
    md_text = ""

    for page in doc:
        text = page.get_text("text")
        md_text += text + "\n\n"

    return md_text

#Hybrid document splitter (Markdown + Recursive)
def markdown_split(md_text):

    headers = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers
    )

    docs = splitter.split_text(md_text)

    return docs

# Semantic Splitter
def semantic_split(docs):

    semantic = SemanticChunker(
        embedding
    )

    final_docs = []

    for doc in docs:

        chunks = semantic.split_text(
            doc.page_content
        )

        final_docs.extend(chunks)

    return final_docs

# STEP 4: Recursive Splitter
def recursive_split(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    final_docs = []

    for doc in docs:
        chunks = splitter.create_documents([doc])
        final_docs.extend(chunks)

    return final_docs


# VECTOR STORE
def create_vector_store(pdf_path):

    os.makedirs("cache", exist_ok=True)
    cache_path = get_cache_path(pdf_path)

    if os.path.exists(cache_path):
        print("Loading embeddings from cache")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Creating new embeddings")

    md = pdf_to_markdown(pdf_path)
    md_docs = markdown_split(md)
    semantic_docs = semantic_split(md_docs)
    final_chunks = recursive_split(semantic_docs)

    embeddings = embed_batches(final_chunks)

    texts = [doc.page_content for doc in final_chunks]

    vector_store = FAISS.from_embeddings(
        list(zip(texts, embeddings)),
        embedding
    )

    with open(cache_path, "wb") as f:
        pickle.dump(vector_store, f)

    return vector_store

def format_docs(docs):

    return "\n\n".join(doc.page_content for doc in docs)

# QA CHAIN
def create_qa_chain(vector_store):

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain