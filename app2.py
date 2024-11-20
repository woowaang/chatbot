import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import urllib3

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 환경 변수 로드
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4")
OPENAI_API_TEMPERATURE = float(os.getenv("OPENAI_API_TEMPERATURE", 0.7))

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key not found. Please set OPENAI_API_KEY in your environment.")

# FastAPI 앱 초기화
app = FastAPI()

# 정적 파일 제공
app.mount("/static", StaticFiles(directory="static"), name="static")

CANCER_INFO_URLS = [
    "https://www.cancer.go.kr/lay1/program/S1T211C212/cancer/view.do?cancer_seq=3341",
    "https://www.cancer.go.kr/lay1/program/S1T211C212/cancer/view.do?cancer_seq=3341&menu_seq=3358",
    "https://www.cancer.go.kr/lay1/program/S1T211C212/cancer/view.do?cancer_seq=3341&menu_seq=3363",
]

def load_and_prepare_documents(urls):
    documents = []
    for url in urls:
        loader = WebBaseLoader(url, requests_kwargs={"verify": False})
        docs = loader.load()
        documents.extend(docs)
    return documents

documents = load_and_prepare_documents(CANCER_INFO_URLS)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(split_docs, embeddings)

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>갑상선암 상담 Chatbot🩺</title>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <h1>갑상선암 상담 Chatbot🩺</h1>
        <div id="chat-box"></div>
        <form id="chat-form">
            <select id="style">
                <option value="친근한 지인☺️">친근한 지인☺️</option>
                <option value="갑상선암 전문가🏥">갑상선암 전문가🏥</option>
            </select>
            <input type="text" id="question" placeholder="상담 내용을 입력해주세요..." autocomplete="off" />
            <button type="button" id="send-btn">Send</button>
        </form>
        <script src="/static/script.js"></script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            question = data["question"]
            style = data["style"]

            if style == "친근한 지인☺️":
                prompt = PromptTemplate(
                    input_variables=["question", "context"],
                    template=(
                        "You are a warm, kind, and empathetic friend or family member. Please answer in Korean. "
                        "Your role is to provide comforting and understandable advice to someone with thyroid cancer. "
                        "Use simple language and maintain a conversational tone, ensuring the person feels supported and cared for.\n\n"
                        "Question: {question}"
                    )
                )
            elif style == "갑상선암 전문가🏥":
                prompt = PromptTemplate(
                    input_variables=["question", "context"],
                    template=(
                        "You are a highly experienced and professional thyroid cancer specialist. Please answer in Korean. "
                        "Use the following pieces of retrieved context to answer the question. "
                        "Your role is to provide detailed, accurate, and evidence-based answers to questions about thyroid cancer, "
                        "including diagnosis, treatment options, post-treatment care, and patient support. "
                        "Maintain a formal and professional tone while delivering precise and reliable information.\n\n"
                        "Context: {context}\nQuestion: {question}"
                    )
                )

            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            relevant_docs = retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            chat = ChatOpenAI(
                model_name=OPENAI_API_MODEL,
                temperature=OPENAI_API_TEMPERATURE,
            )
            chain = LLMChain(llm=chat, prompt=prompt)
            response = chain.run({"question": question, "context": context})

            await websocket.send_json({"question": question, "response": response})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
