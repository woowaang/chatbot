import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import load_prompt
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from app_function import (
    create_faq_search_tool,
    create_tavily_search_tool,
    create_hospital_search_tool,
    create_wikipedia_tool,
)

# 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# LLM 및 도구 초기화
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
tools = [
    create_faq_search_tool(),
    create_tavily_search_tool(),
    create_hospital_search_tool(),
    create_wikipedia_tool(),
]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Streamlit 애플리케이션
def main():
    st.title("💬 갑상선암 상담 Chatbot")

    if "style" not in st.session_state:
        st.session_state.style = "친근한 지인"
    if "history" not in st.session_state:
        st.session_state.history = []

    selected_style = st.selectbox("상담 스타일을 선택하세요", ["친근한 지인", "갑상선암 전문의"])

    if st.session_state.style != selected_style:
        st.session_state.history = []
        st.session_state.style = selected_style
        st.rerun()

    for message in st.session_state.history:
        st.chat_message(message["type"]).write(message["content"])

    user_input = st.chat_input("상담 내용을 입력해 주세요:")
    if user_input:
        with st.chat_message("user"):
            st.session_state.history.append({"type": "user", "content": user_input})
            st.markdown(user_input)

        # 프롬프트 로드
        prompt_file = (
            "친근한지인.yaml" if selected_style == "친근한 지인" else "갑상선암전문의.yaml"
        )
        prompt = load_prompt(prompt_file)

        try:
            context_response = agent.run(user_input)
            final_response = llm.run(prompt.format(question=user_input, agent_scratchpad=context_response))
            st.session_state.history.append({"type": "assistant", "content": final_response})
            st.chat_message("assistant").write(final_response)
        except Exception as e:
            error_message = f"에이전트 실행 중 오류 발생: {e}"
            st.session_state.history.append({"type": "assistant", "content": error_message})
            st.chat_message("assistant").write(error_message)

if __name__ == "__main__":
    main()
