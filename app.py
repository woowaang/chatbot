import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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

# 환경 변수
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4")
OPENAI_API_TEMPERATURE = float(os.getenv("OPENAI_API_TEMPERATURE", 0.7))
USER_AGENT = os.getenv("USER_AGENT", "ThyroidCancerChatbot/1.0")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key not found. Please set OPENAI_API_KEY in your environment.")

# FastAPI 앱 초기화
app = FastAPI()

# 정적 파일 제공
app.mount("/static", StaticFiles(directory="static"), name="static")

# 암 정보 페이지 URL
CANCER_INFO_URLS = [
    "https://www.cancer.go.kr/lay1/program/S1T211C212/cancer/view.do?cancer_seq=3341",
    "https://www.cancer.go.kr/lay1/program/S1T211C212/cancer/view.do?cancer_seq=3341&menu_seq=3358",
    "https://www.cancer.go.kr/lay1/program/S1T211C212/cancer/view.do?cancer_seq=3341&menu_seq=3363",
]

# 데이터 로드 및 벡터화
def load_and_prepare_documents(urls):
    documents = []
    requests_kwargs = {"headers": {"User-Agent": USER_AGENT}, "verify": False}
    for url in urls:
        loader = WebBaseLoader(url, requests_kwargs=requests_kwargs)
        docs = loader.load()
        documents.extend(docs)
    return documents

documents = load_and_prepare_documents(CANCER_INFO_URLS)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

# 벡터 저장소 생성
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(split_docs, embeddings)

# 데이터 모델 정의
class Query(BaseModel):
    question: str
    style: str  # "친근한 지인☺️" 또는 "갑상선암 전문가🏥"

# 루트 경로: HTML 페이지 렌더링
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thyroid Cancer Chatbot</title>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <h1>갑상선암 진단 챗봇🩺</h1>
        <p>상담 스타일을 선택한 후 암에 관련하여 무엇이든 물어보세요.</p>
        <form id="chat-form">
            <label for="question">Question:</label><br>
            <input type="text" id="question" name="question" placeholder="Enter your question here..."><br><br>
            <label for="style">Style:</label><br>
            <select id="style" name="style">
                <option value="친근한 지인☺️">친근한 지인☺️</option>
                <option value="갑상선암 전문가🏥">갑상선암 전문가🏥</option>
            </select><br><br>
            <button type="button" id="submit-btn">Submit</button>
        </form>
        <h2>Response:</h2>
        <div id="response"></div>
        <script src="/static/script.js"></script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Chat API 엔드포인트
@app.post("/chat")
async def chat(query: Query):
    try:
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        # 스타일에 따른 프롬프트 정의
        if query.style == "친근한 지인☺️":
            prompt = PromptTemplate(
                input_variables=["question", "context"],
                template=(
                    "You are a warm, kind, and empathetic friend or family member. Please answer in Korean. "
                    "Your role is to provide comforting and understandable advice to someone with thyroid cancer. "
                    "Use simple language and maintain a conversational tone, ensuring the person feels supported and cared for.\n\n"
                    "Question: {question}"
                )
            )
        elif query.style == "갑상선암 전문가🏥":
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
        else:
            raise HTTPException(status_code=400, detail="Invalid style selected.")

        # 문서 검색 및 컨텍스트 생성
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(query.question)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # LLM 사용하여 응답 생성
        chat = ChatOpenAI(
            model_name=OPENAI_API_MODEL,
            temperature=OPENAI_API_TEMPERATURE,
        )
        chain = LLMChain(llm=chat, prompt=prompt)
        response = chain.run({"question": query.question, "context": context})

        return {"response": response}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
