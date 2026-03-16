# Part 5 체크포인트

## 5-1. 텍스트 분류
- [ ] HuggingFace Pipeline으로 감정분석을 실행할 수 있다
- [ ] BERT Fine-tuning의 전체 과정 (데이터→토크나이저→모델→학습→평가)을 설명할 수 있다
- [ ] BERT Fine-tuning 코드를 처음부터 혼자 작성할 수 있다
- [ ] Accuracy, F1, Confusion Matrix의 차이를 설명할 수 있다
- [ ] 한국어 모델 (KoBERT 등)을 사용할 수 있다

## 5-2. 챗봇 웹사이트
- [ ] OpenAI 또는 Claude API를 호출하는 코드를 작성할 수 있다
- [ ] System Prompt, Temperature, Token의 역할을 설명할 수 있다
- [ ] Streamlit으로 챗봇 웹 UI를 만들 수 있다
- [ ] Gradio로 데모 페이지를 만들 수 있다
- [ ] 스트리밍 응답을 구현할 수 있다

## 5-3. YOLO
- [ ] Detection 발전사 (R-CNN → Fast → Faster → Mask → YOLO) 를 설명할 수 있다
- [ ] Two-Stage vs One-Stage의 차이를 설명할 수 있다
- [ ] YOLOv8 사전학습 모델로 이미지/웹캠 탐지를 실행할 수 있다
- [ ] Roboflow로 데이터 라벨링 → YOLO format 변환을 할 수 있다
- [ ] 커스텀 데이터셋으로 YOLO Fine-tuning을 할 수 있다
- [ ] mAP, Precision, Recall의 의미를 설명할 수 있다

## 코딩 목표 수치

| 태스크 | 목표 | 달성 |
|--------|------|------|
| IMDB 감정분류 accuracy | ≥ 88% | [ ] ___% |
| 네이버 영화 리뷰 accuracy | ≥ 85% | [ ] ___% |
| 커스텀 데이터 YOLO mAP50 | ≥ 0.7 | [ ] ___ |

## 실전 테스트 — 과제 투입 가능 확인

| # | 테스트 | 제한 시간 | 합격 기준 | 통과 |
|---|--------|----------|----------|------|
| 1 | 새 텍스트 데이터셋 → BERT 분류기 학습 + 평가 | 30분 | F1 ≥ 0.8 | [ ] |
| 2 | API 챗봇 웹사이트 완성 + 동작 확인 | 1시간 | 대화 가능 | [ ] |
| 3 | 새 이미지 50장 라벨링 + YOLO 학습 + 웹캠 탐지 | 2시간 | 탐지 동작 | [ ] |
