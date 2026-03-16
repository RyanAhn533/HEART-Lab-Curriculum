# Part 5: 실전 응용

> 원리를 이해했으니, 이제 과제에 바로 투입될 수 있는 도구를 익힌다.
> 목표: "이론을 알아" 수준이 아니라 **"혼자서 만들 수 있어"** 수준.

## 폴더 구조

```
Part5_practical/
├── 5-1_text_classification/
│   └── 01_text_classification_huggingface.py   ← BERT Fine-tuning
├── 5-2_chatbot_web/
│   ├── 01_chatbot_streamlit.py                 ← Streamlit 챗봇
│   └── 02_chatbot_gradio.py                    ← Gradio 챗봇
└── 5-3_yolo/
    ├── 01_yolo_quickstart.py                   ← YOLOv8 사전학습, 웹캠
    └── 02_yolo_custom_training.py              ← 커스텀 학습
```

## 합격 기준

| 테스트 | 제한 시간 | 기준 |
|--------|----------|------|
| 새 텍스트 → BERT 분류기 학습+평가 | 30분 | F1 ≥ 0.8 |
| API 챗봇 웹사이트 완성 | 1시간 | 대화 가능 |
| 이미지 50장 라벨링 → YOLO 학습 → 웹캠 탐지 | 2시간 | 탐지 동작 |
