# Part 3: 데이터 유형별 모델

> MLP만으로는 부족하다. 데이터 형태에 따라 더 잘하는 모델이 있다.

## 왜 모델이 여러 개 필요한가?

```
MLP: 입력을 일렬로 펴서 처리
  → 이미지의 공간 관계? 모름 → CNN 필요
  → 시계열의 순서 관계? 모름 → RNN 필요
  → 긴 시퀀스의 앞 정보? 잊어버림 → LSTM 필요
```

## 폴더 구조

```
Part3_data_models/
├── 3-1_cnn/
│   └── 01_convolution_numpy.py    ← Conv 연산, Pooling, 출력 크기 공식
├── 3-2_rnn/
│   └── 01_rnn_numpy.py            ← Hidden State, Vanishing Gradient
└── 3-3_lstm/
    └── 01_lstm_gate_numpy.py      ← 3개 Gate, Cell State
```

## 학습 순서

1. **3-1 CNN**: 이미지 → 필터로 공간 특징 추출
2. **3-2 RNN**: 시계열/텍스트 → Hidden State로 순서 기억
3. **3-3 LSTM**: RNN의 기억력 한계 → Gate로 해결

각 모델 시작: "이전 모델의 어떤 한계 때문에 이게 나왔는가?"
