"""
========================================
10-02. YOLO 커스텀 데이터셋 학습
========================================
내 데이터로 YOLO 학습시키기!

이 파일에서 배우는 것:
1. 데이터 라벨링 (Roboflow / LabelImg)
2. 데이터셋 구조 (YOLO format)
3. YOLOv8 Fine-tuning
4. 학습 결과 평가 (mAP, Precision, Recall)

실제 프로젝트 예시: 차량 내부 운전자 감정/행동 인식
"""

from ultralytics import YOLO
import os

# ==============================================
# 1단계: 데이터셋 구조
# ==============================================
print("=" * 60)
print("1. YOLO 데이터셋 구조")
print("=" * 60)
print("""
데이터셋 폴더 구조:
  dataset/
  ├── train/
  │   ├── images/        ← 학습 이미지 (.jpg, .png)
  │   │   ├── img001.jpg
  │   │   └── img002.jpg
  │   └── labels/        ← 라벨 파일 (.txt)
  │       ├── img001.txt
  │       └── img002.txt
  ├── val/
  │   ├── images/
  │   └── labels/
  └── data.yaml          ← 데이터셋 설정 파일

라벨 파일 형식 (YOLO format):
  class_id  x_center  y_center  width  height
  0         0.5       0.4       0.3    0.6
  → 모든 값은 이미지 크기 대비 비율 (0~1)

라벨링 도구:
  - Roboflow (roboflow.com) ← 추천! 웹에서 바로 가능
  - LabelImg (로컬 도구)
  - CVAT (온라인 도구)
""")

# ==============================================
# 2단계: data.yaml 작성
# ==============================================
print("=" * 60)
print("2. data.yaml 작성 예시")
print("=" * 60)

yaml_content = """
# data.yaml
path: ./dataset           # 데이터셋 경로
train: train/images        # 학습 이미지 경로
val: val/images            # 검증 이미지 경로

# 클래스 정의
names:
  0: person
  1: car
  2: truck
  3: bicycle
"""
print(yaml_content)

# 예시 데이터셋 다운로드 (Roboflow 공개 데이터셋)
print("Roboflow에서 데이터셋 다운로드 예시:")
print("""
  from roboflow import Roboflow
  rf = Roboflow(api_key="YOUR_API_KEY")
  project = rf.workspace("your-workspace").project("your-project")
  version = project.version(1)
  dataset = version.download("yolov8")
""")

# ==============================================
# 3단계: 학습
# ==============================================
print("\n" + "=" * 60)
print("3. YOLOv8 커스텀 학습")
print("=" * 60)

def train_custom_yolo():
    """커스텀 데이터셋으로 YOLO 학습"""
    # 사전학습 모델 로드
    model = YOLO("yolov8n.pt")  # nano 모델로 시작 (빠르게 실험)

    # 학습 시작
    results = model.train(
        data="data.yaml",        # 데이터셋 설정
        epochs=50,               # 에폭 수
        imgsz=640,               # 입력 이미지 크기
        batch=16,                # 배치 크기
        patience=10,             # Early stopping
        save=True,               # 모델 저장
        device=0,                # GPU 사용 (CPU는 "cpu")
        workers=4,               # 데이터 로더 워커 수
        project="runs/custom",   # 결과 저장 경로
        name="exp1",             # 실험 이름
    )

    return results

# 학습 실행 (data.yaml이 준비된 후 주석 해제)
# results = train_custom_yolo()

# ==============================================
# 4단계: 학습 결과 평가
# ==============================================
print("\n" + "=" * 60)
print("4. 결과 평가 및 추론")
print("=" * 60)

def evaluate_and_predict():
    """학습된 모델 평가 및 추론"""
    # 학습된 모델 로드
    model = YOLO("runs/custom/exp1/weights/best.pt")

    # 검증 데이터 평가
    metrics = model.val()
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    # 새 이미지에 추론
    results = model("test_image.jpg")
    results[0].save(filename="custom_detection.jpg")
    print("결과 저장: custom_detection.jpg")

# evaluate_and_predict()

# ==============================================
# 5단계: Detection 모델 발전 역사 정리
# ==============================================
print("\n" + "=" * 60)
print("5. Object Detection 발전 역사")
print("=" * 60)
print("""
┌──────────────┬──────┬───────────────────────────────────────┐
│ 모델          │ 년도  │ 핵심 아이디어                          │
├──────────────┼──────┼───────────────────────────────────────┤
│ R-CNN        │ 2014 │ Selective Search → CNN → SVM          │
│              │      │ 느림 (이미지당 ~47초)                   │
├──────────────┼──────┼───────────────────────────────────────┤
│ Fast R-CNN   │ 2015 │ 이미지 전체를 한번에 CNN 통과           │
│              │      │ RoI Pooling으로 영역 추출               │
├──────────────┼──────┼───────────────────────────────────────┤
│ Faster R-CNN │ 2015 │ Region Proposal Network (RPN) 도입     │
│              │      │ End-to-end 학습 가능                   │
├──────────────┼──────┼───────────────────────────────────────┤
│ Mask R-CNN   │ 2017 │ + Instance Segmentation (픽셀 단위)    │
│              │      │ Detection + Segmentation 동시에        │
├──────────────┼──────┼───────────────────────────────────────┤
│ YOLO v1      │ 2016 │ "You Only Look Once" - 1단계 검출      │
│              │      │ 실시간 가능! (~45 FPS)                  │
├──────────────┼──────┼───────────────────────────────────────┤
│ YOLOv8       │ 2023 │ Anchor-free, 최적화된 아키텍처          │
│              │      │ 분류/탐지/세그멘테이션/포즈 통합         │
├──────────────┼──────┼───────────────────────────────────────┤
│ YOLOv11      │ 2024 │ 최신 모델, C2PSA, 효율성 향상           │
└──────────────┴──────┴───────────────────────────────────────┘

Two-Stage (정확): R-CNN 계열 → 영역 제안 + 분류 (2단계)
One-Stage (빠름): YOLO, SSD → 한번에 검출 (1단계)
""")

# ==============================================
# ★ 과제 ★
# ==============================================
"""
[실습 과제 - 실전 프로젝트]
1. Roboflow에서 데이터셋을 만들거나 다운받아 YOLO 학습을 해보세요.
   추천: 안전모 탐지, 마스크 탐지, 차량 번호판 탐지

2. 학습 후 웹캠 실시간 탐지를 해보세요.

3. mAP, Precision, Recall의 의미를 설명하고 confusion matrix를 분석하세요.

4. 같은 데이터셋으로 yolov8n vs yolov8m vs yolov8l 성능을 비교하세요.

5. YOLO의 export 기능으로 ONNX/TensorRT로 변환하고 추론 속도를 비교하세요.
   model.export(format="onnx")
"""
