"""
========================================
02-01. Convolution 연산 - numpy 손코딩
========================================
CNN의 핵심: 이미지에 필터(커널)를 슬라이딩하며 특징 추출

이 파일에서 배우는 것:
1. 2D Convolution 연산을 손으로 계산
2. 필터가 어떤 특징을 잡는지 시각화
3. Padding, Stride 개념
4. Pooling (MaxPool, AvgPool)

★ 3x3 이미지에 2x2 필터 적용을 종이에 먼저 계산 ★
"""

import numpy as np

# ==============================================
# 1단계: 가장 간단한 Convolution
# ==============================================
print("=" * 50)
print("2D Convolution 손계산")
print("=" * 50)

# 5x5 입력 이미지 (숫자가 밝기)
image = np.array([
    [1, 2, 0, 1, 3],
    [0, 1, 2, 3, 1],
    [1, 0, 1, 0, 2],
    [2, 1, 0, 1, 0],
    [0, 3, 2, 1, 1]
], dtype=float)

# 3x3 필터 (Edge Detection - 수직 엣지)
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=float)

print(f"입력 이미지 (5x5):\n{image}\n")
print(f"필터 (3x3) - 수직 엣지 검출:\n{kernel}\n")

def conv2d(image, kernel, stride=1, padding=0):
    """2D Convolution 직접 구현"""
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)

    h, w = image.shape
    kh, kw = kernel.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # 필터 영역 추출
            region = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            # Element-wise 곱 후 합산
            output[i, j] = np.sum(region * kernel)

    return output

"""
★★★ 손계산 가이드 ★★★

위치 (0,0)의 계산:
  region = image[0:3, 0:3] = [[1,2,0], [0,1,2], [1,0,1]]
  output[0,0] = sum(region * kernel)
             = 1*(-1) + 2*0 + 0*1 + 0*(-1) + 1*0 + 2*1 + 1*(-1) + 0*0 + 1*1
             = -1 + 0 + 0 + 0 + 0 + 2 + -1 + 0 + 1
             = 1

직접 계산해서 코드 결과와 비교하세요!
"""

# stride=1, padding=0
result = conv2d(image, kernel)
print(f"Convolution 결과 (stride=1, no padding):")
print(f"출력 크기: {result.shape} = ({image.shape[0]}-{kernel.shape[0]})/{1}+1 = 3")
print(f"{result}\n")

# stride=1, padding=1 (same padding)
result_pad = conv2d(image, kernel, padding=1)
print(f"Convolution 결과 (stride=1, padding=1):")
print(f"출력 크기: {result_pad.shape} (입력과 같은 크기 유지)")
print(f"{result_pad}\n")

# stride=2
result_s2 = conv2d(image, kernel, stride=2)
print(f"Convolution 결과 (stride=2):")
print(f"출력 크기: {result_s2.shape}")
print(f"{result_s2}\n")

# ==============================================
# 2단계: MaxPooling
# ==============================================
print("=" * 50)
print("MaxPooling")
print("=" * 50)

def maxpool2d(image, pool_size=2, stride=2):
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = image[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[i, j] = np.max(region)

    return output

test_feature = np.array([
    [1, 3, 2, 0],
    [0, 5, 1, 2],
    [3, 1, 0, 4],
    [2, 0, 3, 1]
], dtype=float)

print(f"입력 feature map (4x4):\n{test_feature}\n")
pooled = maxpool2d(test_feature)
print(f"MaxPool 결과 (2x2, stride=2):\n{pooled}")
print("→ 각 2x2 영역에서 최댓값만 추출")

# ==============================================
# 3단계: 실제 이미지에 다양한 필터 적용
# ==============================================
print("\n" + "=" * 50)
print("다양한 필터 효과")
print("=" * 50)

filters = {
    "수직 엣지": np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=float),
    "수평 엣지": np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=float),
    "샤프닝":   np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float),
    "블러":     np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=float) / 9,
}

for name, f in filters.items():
    result = conv2d(image, f, padding=1)
    print(f"\n{name} 필터:\n{f}")
    print(f"결과:\n{result}")

# ==============================================
# 4단계: 출력 크기 공식 (매우 중요!)
# ==============================================
print("\n" + "=" * 50)
print("출력 크기 공식")
print("=" * 50)
print("""
Output Size = (Input - Kernel + 2*Padding) / Stride + 1

예시:
  Input=28, Kernel=3, Padding=0, Stride=1 → (28-3+0)/1+1 = 26
  Input=28, Kernel=3, Padding=1, Stride=1 → (28-3+2)/1+1 = 28 (same)
  Input=28, Kernel=3, Padding=0, Stride=2 → (28-3+0)/2+1 = 13
  Input=28, Kernel=5, Padding=2, Stride=1 → (28-5+4)/1+1 = 28 (same)

★ 과제에서 이 공식을 직접 적용해보세요 ★
""")

# ==============================================
# ★ 과제 ★
# ==============================================
"""
[손계산 과제]
1. 4x4 이미지에 3x3 필터를 stride=1, padding=0으로 적용했을 때
   출력의 각 값을 종이에 계산하세요.

2. 같은 이미지에 2x2 MaxPooling (stride=2)을 적용한 결과를 계산하세요.

3. 출력 크기 공식으로 다음을 계산:
   (a) Input=32x32, Kernel=5x5, Padding=0, Stride=1
   (b) Input=224x224, Kernel=7x7, Padding=3, Stride=2

[코딩 과제]
4. MNIST 이미지(28x28)를 불러와서 위의 필터들을 적용하고 시각화하세요.
5. Conv → ReLU → MaxPool → Conv → ReLU → MaxPool 파이프라인을
   numpy로 구현해보세요.
"""
