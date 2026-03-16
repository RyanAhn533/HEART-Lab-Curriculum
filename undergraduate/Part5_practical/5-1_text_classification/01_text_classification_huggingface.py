"""
========================================
08-01. 텍스트 분류 (Transformer 기반)
========================================
Transformer를 배웠으니 실제로 써보자!

이 파일에서 배우는 것:
1. HuggingFace Transformers 라이브러리 사용법
2. 사전학습 모델(BERT) 로드 및 Fine-tuning
3. 감정 분석 (Sentiment Analysis)
4. 뉴스 분류 (Text Classification)

파이프라인: 데이터 → 토크나이저 → 모델 → 학습 → 평가
"""

# ==============================================
# 1단계: 바로 써보기 (Pipeline)
# ==============================================
print("=" * 60)
print("1. HuggingFace Pipeline으로 바로 감정분석")
print("=" * 60)

from transformers import pipeline

# 한 줄로 감정분석!
classifier = pipeline("sentiment-analysis")

texts = [
    "I love this movie, it was fantastic!",
    "This is the worst product I have ever bought.",
    "The weather is okay today.",
    "I'm so happy to be here!",
]

print("영어 감정 분석 결과:")
for text in texts:
    result = classifier(text)[0]
    print(f"  \"{text}\"")
    print(f"    → {result['label']} (confidence: {result['score']:.4f})\n")

# ==============================================
# 2단계: BERT로 커스텀 데이터 분류
# ==============================================
print("=" * 60)
print("2. BERT Fine-tuning - IMDB 감정 분류")
print("=" * 60)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

# 데이터셋 로드 (IMDB 영화 리뷰)
print("IMDB 데이터셋 로딩...")
dataset = load_dataset("imdb")
# 빠른 실험을 위해 일부만 사용
small_train = dataset["train"].shuffle(seed=42).select(range(1000))
small_test = dataset["test"].shuffle(seed=42).select(range(200))

print(f"학습 데이터: {len(small_train)} 개")
print(f"테스트 데이터: {len(small_test)} 개")
print(f"\n예시:")
print(f"  리뷰: {small_train[0]['text'][:100]}...")
print(f"  라벨: {small_train[0]['label']} (0=부정, 1=긍정)")

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

print("\n토크나이징...")
train_dataset = small_train.map(tokenize_function, batched=True)
test_dataset = small_test.map(tokenize_function, batched=True)

# 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("\nBERT Fine-tuning 시작...")
trainer.train()

# 평가
results = trainer.evaluate()
print(f"\n테스트 정확도: {results['eval_accuracy']:.4f}")

# ==============================================
# 3단계: 학습된 모델로 예측
# ==============================================
print("\n" + "=" * 60)
print("3. 학습된 모델로 직접 예측")
print("=" * 60)

import torch

model.eval()
test_texts = [
    "This movie is absolutely brilliant and touching!",
    "Terrible acting, boring plot, waste of time.",
    "It was an average film, nothing special.",
]

for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()
    label = "긍정" if pred == 1 else "부정"
    print(f"  \"{text}\"")
    print(f"    → {label} (부정: {probs[0][0]:.3f}, 긍정: {probs[0][1]:.3f})\n")

# ==============================================
# ★ 과제 ★
# ==============================================
"""
[실습 과제]
1. 한국어 감정 분석: "monologg/kobert"로 네이버 영화 리뷰 분류하세요.
2. 뉴스 분류: AG News 데이터셋으로 4개 카테고리 분류하세요.
3. 학습 데이터 수를 100, 500, 1000, 5000으로 바꿔가며 정확도 변화를 관찰하세요.
4. BERT 대신 DistilBERT로 바꾸면 속도/정확도가 어떻게 변하나요?
5. epoch을 1, 2, 5, 10으로 바꿔가며 과적합(overfitting)을 관찰하세요.
"""
