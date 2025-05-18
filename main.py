from fastapi import FastAPI, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from chat import get_chat_response
from utils.rag_chain import call_plain_llm
from index_documents import index_pdf_document
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import os, uuid

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# In-memory history store (you can persist this if needed)
chat_history_store = {}
class ChatRequest(BaseModel):
    query: str
    mode: str = "rag-llm"

def generate_session_id():
    return str(uuid.uuid4())

def get_or_create_session_id(request: Request, response: Response) -> str:
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = generate_session_id()
        response.set_cookie(key="session_id", value=session_id, httponly=True)
    return session_id

async def process_uploaded_file(file: UploadFile, session_id: str):
    os.makedirs("temp_files", exist_ok=True)
    file_path = f"temp_files/{session_id}_{file.filename}"
    content = await file.read() 
    with open(file_path, "wb") as f:
        f.write(content)
    index_pdf_document(file_path, session_id) 

@app.post("/upload")
async def upload_file(
    request: Request,
    response: Response,
    file: UploadFile = File(...)
):
    session_id = get_or_create_session_id(request, response)
    await process_uploaded_file(file, session_id)  
    return JSONResponse(
        content={"success": True, "session_id": session_id},
        headers={"Set-Cookie": f"session_id={session_id}; HttpOnly; Path=/"}
    )


@app.post("/chat")
async def chat(request: Request, response: Response, body: ChatRequest):
    session_id = get_or_create_session_id(request, response)
    history = chat_history_store.get(session_id, [])
    query = body.query
    mode = body.mode
    if mode == "llm":
        response = call_plain_llm(query, history)
        updated_history = history  # <-- ADD THIS
    else:
        response, updated_history = get_chat_response(query, session_id, history)

    chat_history_store[session_id] = updated_history
    return JSONResponse(content={"response": response})

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    response = templates.TemplateResponse("index.html", {"request": request})
    if not request.cookies.get("session_id"):
        response.set_cookie("session_id", str(uuid.uuid4()))
    return response

