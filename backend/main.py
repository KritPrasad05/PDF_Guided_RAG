import sys
sys.stdout = sys.stderr

import tempfile
import atexit
import shutil

print("MAIN.PY LOADED")
import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from rag_pipeline import create_vector_store, create_qa_chain
import traceback

app = FastAPI()
qa_chain = None

# Create temporary upload folder
UPLOAD_DIR = tempfile.mkdtemp(prefix="rag_upload_")

# Auto delete uploads when app closes
def cleanup_uploads():

    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)

atexit.register(cleanup_uploads)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        print("UPLOAD FUNCTION STARTED")
        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )
        content = await file.read()          # ← fixed file read
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        print("PDF SAVED")

        vector_store = create_vector_store(file_path)
        print("VECTOR CREATED")

        global qa_chain
        qa_chain = create_qa_chain(vector_store)
        print("CHAIN CREATED")

        return {"status": "success"}

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()        # ← this prints the full stack trace
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask")
def ask_question(question: str):

    global qa_chain

    if qa_chain is None:

        raise HTTPException(
            status_code=400,
            detail="Upload a PDF first"
        )

    answer = qa_chain.invoke(question)

    return {
        "answer": answer
    }
