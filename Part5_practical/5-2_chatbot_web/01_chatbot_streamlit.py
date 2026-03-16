"""
========================================
09-01. AI 챗봇 웹사이트 (Streamlit + API)
========================================
LLM API를 불러다가 간단한 챗봇 웹사이트를 만듭니다.

이 파일에서 배우는 것:
1. OpenAI API / Claude API 사용법
2. Streamlit으로 웹 UI 만들기
3. 대화 히스토리 관리
4. System Prompt 활용

실행: streamlit run 01_chatbot_streamlit.py
"""

import streamlit as st

# ==============================================
# 설정 (아래 중 하나 선택)
# ==============================================
# .env 파일이나 환경변수에 API KEY를 설정하세요
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

API_PROVIDER = "openai"  # "openai" 또는 "anthropic"

# ==============================================
# API 클라이언트 초기화
# ==============================================
def get_ai_response(messages, system_prompt=""):
    """AI API를 호출하여 응답을 받습니다."""

    if API_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()  # OPENAI_API_KEY 환경변수 자동 사용

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 저렴한 모델부터 시작
            messages=full_messages,
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content

    elif API_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경변수 자동 사용

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

# ==============================================
# Streamlit UI
# ==============================================
st.set_page_config(page_title="HEART Lab AI Chatbot", page_icon="🤖", layout="wide")

st.title("AI Chatbot")
st.caption(f"API: {API_PROVIDER}")

# 사이드바: 설정
with st.sidebar:
    st.header("설정")

    system_prompt = st.text_area(
        "System Prompt",
        value="당신은 친절한 AI 어시스턴트입니다. 한국어로 답변합니다.",
        height=100,
    )

    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **사용법:**
    1. API Key를 환경변수에 설정
    2. System Prompt로 AI 성격 설정
    3. 메시지 입력 후 전송

    **예시 System Prompt:**
    - "너는 파이썬 전문가야. 코드를 알려줘."
    - "너는 영어 선생님이야. 문법을 교정해줘."
    - "너는 감정 분석기야. 문장의 감정을 분석해."
    """)

# 대화 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # AI 응답
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                response = get_ai_response(
                    st.session_state.messages,
                    system_prompt=system_prompt,
                )
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"API 오류: {e}")
                st.info("API Key가 환경변수에 설정되어 있는지 확인하세요.")

# ==============================================
# ★ 과제 ★
# ==============================================
"""
[실습 과제]
1. System Prompt를 바꿔가며 AI 성격을 다르게 만들어보세요.
   - "너는 셰익스피어야. 모든 답변을 고풍스러운 영어로 해."
   - "너는 코드 리뷰어야. 코드를 보고 개선점을 알려줘."

2. Gradio로 같은 챗봇을 만들어보세요 (gradio.ChatInterface 사용).

3. 파일 업로드 기능을 추가하여 PDF/텍스트 파일을 읽고 질문에 답하는
   RAG 챗봇을 만들어보세요.

4. 채팅 히스토리를 파일로 저장/로드하는 기능을 추가하세요.

5. 스트리밍 응답 (한 글자씩 출력) 기능을 구현하세요.
   (OpenAI: stream=True, Anthropic: stream으로 변경)
"""
