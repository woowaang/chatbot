import os
import json
import requests
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import Tool
from langchain.tools import WikipediaAPIWrapper
from langchain_community.retrievers import TavilySearchAPIRetriever

# FAQ 검색 도구 생성
def create_faq_search_tool():
    # FAQ 데이터 로드
    with open("faq_data.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    # FAQ 데이터를 Document 형식으로 변환
    documents = [
        Document(page_content=f"질문: {item['question']}\n답변: {item['answer']}")
        for item in faq_data
    ]

    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    # 벡터 저장소 생성
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # FAQ 검색 함수
    def search_with_faiss(query: str) -> str:
        """FAQ 데이터 검색"""
        results = vectorstore.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in results])

    # Tool 반환
    return Tool(
        name="FAQ Search",
        func=search_with_faiss,
        description="FAQ 데이터에서 질문과 유사한 항목을 검색하여 적절한 답변을 제공합니다.",
    )

# Tavily 검색 도구 생성
def create_tavily_search_tool():
    def tavily_search(query: str) -> str:
        """Tavily API 검색"""
        try:
            retriever = TavilySearchAPIRetriever(k=3)
            docs = retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs if hasattr(doc, "page_content")])
        except Exception as e:
            return f"Tavily 검색 도중 오류 발생: {e}"

    return Tool(
        name="Tavily Search",
        func=tavily_search,
        description="Tavily API를 사용해 웹에서 최신 정보를 검색합니다.",
    )

# 병원 검색 도구 생성
def create_hospital_search_tool():
    def hospital_search(location: str) -> str:
        """Google Maps API를 사용해 병원 검색"""
        try:
            api_key = os.getenv("GOOGLE_MAPS_API_KEY")
            url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {
                "query": f"갑상선암 병원 in {location}",
                "key": api_key,
                "language": "ko",
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get("results", [])
            return "\n".join([f"{result['name']} - {result['formatted_address']}" for result in results[:5]])
        except Exception as e:
            return f"병원 검색 도중 오류 발생: {e}"

    return Tool(
        name="Nearby Thyroid Cancer Hospitals",
        func=hospital_search,
        description="사용자의 위치를 기반으로 갑상선암 병원을 검색합니다.",
    )

# Wikipedia 검색 도구 생성
def create_wikipedia_tool():
    wikipedia_tool = Tool(
        name="Wikipedia Search",
        func=WikipediaAPIWrapper().run,
        description="Use this tool to fetch information from Wikipedia articles."
    )
    return wikipedia_tool
