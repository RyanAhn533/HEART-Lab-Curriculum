# Part 4: 현대 아키텍처

> LSTM → Seq2Seq → Attention → Transformer. 각 모델이 이전의 한계를 해결한다.

## 발전 흐름

```
LSTM: 입출력 길이가 같아야 함
  → Seq2Seq: Encoder-Decoder로 해결, 하지만 Context Vector 병목
    → Attention: 모든 timestep 참조로 해결, 하지만 순차처리라 느림
      → Transformer: Attention만으로 전부, 병렬 처리 가능!
```

## 폴더 구조

```
Part4_modern/
├── 4-1_seq2seq/
│   └── 01_seq2seq_numpy.py           ← Encoder-Decoder, Context Vector
├── 4-2_attention/
│   └── 01_attention_numpy.py         ← QKV, Attention Heatmap
└── 4-3_transformer/
    └── 01_self_attention_numpy.py    ← Self-Attention, Multi-Head, PE
```
