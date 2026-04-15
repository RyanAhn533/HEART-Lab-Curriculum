# Part 4 체크포인트

## 4-1. Seq2Seq
- [ ] 입출력 길이가 다른 문제에 LSTM만으로 부족한 이유를 설명할 수 있다
- [ ] Encoder → Context Vector → Decoder 흐름을 그림으로 그릴 수 있다
- [ ] Teacher Forcing이 뭔지 설명할 수 있다
- [ ] Context Vector 병목 문제를 설명할 수 있다

## 4-2. Attention
- [ ] Seq2Seq의 Context Vector 병목을 Attention이 어떻게 해결하는지 설명할 수 있다
- [ ] Attention Score → softmax → Weight → Context 과정을 설명할 수 있다
- [ ] 3개 토큰에 대한 Attention을 손으로 계산할 수 있다
- [ ] Attention Heatmap이 뭘 의미하는지 설명할 수 있다

## 4-3. Transformer
- [ ] RNN 기반 Attention의 한계 (순차처리)를 설명할 수 있다
- [ ] Self-Attention이 RNN과 다른 점을 설명할 수 있다
- [ ] Q, K, V 행렬곱 → Scaled Dot-Product 를 손으로 계산할 수 있다
- [ ] Multi-Head Attention의 필요성을 설명할 수 있다
- [ ] Positional Encoding이 왜 필요한지 설명할 수 있다
- [ ] Transformer Block 전체 구조를 **보지 않고** 그릴 수 있다

## 손코딩 시험

| # | 문제 | 목표 시간 | 통과 |
|---|------|----------|------|
| 1 | Seq2Seq Encoder 3 timestep → Context Vector | 10분 | [ ] |
| 2 | Attention Score 3개 토큰 + softmax | 10분 | [ ] |
| 3 | Self-Attention Q, K, V 전체 계산 | 15분 | [ ] |
| 4 | Transformer Block 구조 그리기 | 보지 않고 | [ ] |

## 코딩 목표 수치

| 태스크 | 목표 | 달성 |
|--------|------|------|
| Seq2Seq 숫자 뒤집기 accuracy | ≥ 90% | [ ] ___% |

## "왜 이 모델이 등장했는지" (자기 말로 작성)

| 모델 | 이전 모델의 어떤 한계를 | 어떻게 해결했는지 |
|------|---------------------|-----------------|
| Seq2Seq | | |
| Attention | | |
| Transformer | | |
