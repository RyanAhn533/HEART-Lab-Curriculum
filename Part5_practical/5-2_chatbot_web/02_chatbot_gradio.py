"""
========================================
09-02. AI 챗봇 (Gradio 버전)
========================================
Gradio는 ML 데모를 만드는 가장 빠른 방법입니다.
설치: pip install gradio openai

실행: python 02_chatbot_gradio.py
→ 자동으로 브라우저가 열리고 http://localhost:7860 에서 실행
"""

import gradio as gr
from openai import OpenAI

client = OpenAI()  # OPENAI_API_KEY 환경변수


def chat(message, history, system_prompt):
    """대화 함수"""
    messages = [{"role": "system", "content": system_prompt}]

    # 이전 대화 히스토리 추가
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    # 스트리밍 응답
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )

    partial = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial += chunk.choices[0].delta.content
            yield partial  # 한 글자씩 출력!


# Gradio UI
demo = gr.ChatInterface(
    fn=chat,
    title="HEART Lab AI Chatbot (Gradio)",
    description="AI 챗봇 데모입니다. System Prompt를 바꿔가며 실험해보세요.",
    additional_inputs=[
        gr.Textbox(
            value="당신은 친절한 AI 어시스턴트입니다. 한국어로 답변합니다.",
            label="System Prompt",
            lines=3,
        ),
    ],
    examples=[
        ["파이썬에서 리스트와 튜플의 차이가 뭐야?"],
        ["Transformer의 Self-Attention을 설명해줘"],
        ["YOLO 모델의 장점은?"],
    ],
    theme="soft",
)

if __name__ == "__main__":
    demo.launch(share=False)  # share=True로 하면 공유 링크 생성
