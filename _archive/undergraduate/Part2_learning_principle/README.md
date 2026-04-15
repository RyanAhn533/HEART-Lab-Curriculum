# Part 2: 학습의 원리

> AI = 오차를 줄이는 과정. 그 "줄이는 방법"을 가장 단순한 모델부터 손으로 직접 해본다.

## 폴더 구조

```
Part2_learning_principle/
├── 2-1_linear_regression/
│   └── 01_linear_regression_numpy.py   ← y=wx+b, 10 epoch 손계산 검증
└── 2-2_ann/
    ├── 01_perceptron_numpy.py          ← 단일 뉴런, AND/OR/XOR
    ├── 02_mlp_backprop_numpy.py        ← MLP Chain Rule 역전파 손코딩
    └── 03_mlp_keras_pytorch.py         ← 같은 것을 프레임워크로 비교
```

## 학습 순서

1. **2-1 선형회귀**: Forward → Loss → Gradient → Update 를 10 epoch 손계산
2. **2-2 ANN**: 층을 쌓고 Chain Rule로 역전파, numpy → Keras/PyTorch 비교

## 핵심

```
모든 AI 모델의 학습:
Input → Model → Prediction → Loss(pred, y) → Backward(Chain Rule) → Weight Update

이 파이프라인을 손으로 직접 계산한 사람만이 "프레임워크가 뭘 해주는지" 안다.
```
