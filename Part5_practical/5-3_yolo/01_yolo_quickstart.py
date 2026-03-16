"""
========================================
10-01. YOLO 실시간 객체 탐지
========================================
YOLOv8로 실시간 Object Detection!

설치: pip install ultralytics opencv-python

이 파일에서 배우는 것:
1. YOLOv8 사전학습 모델 사용 (COCO 80 클래스)
2. 이미지에서 객체 탐지
3. 웹캠 실시간 탐지
4. 커스텀 데이터셋 학습

Detection 파이프라인:
  Image → Backbone(특징추출) → Neck(FPN) → Head(검출) → NMS → Boxes
"""

from ultralytics import YOLO
import cv2
import numpy as np

# ==============================================
# 1단계: 사전학습 모델로 즉시 사용
# ==============================================
print("=" * 60)
print("1. YOLOv8 사전학습 모델")
print("=" * 60)

# 모델 다운로드 & 로드 (자동)
model = YOLO("yolov8n.pt")  # nano (가장 가벼움)
# 다른 옵션: yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (extra)

print(f"모델: YOLOv8n")
print(f"클래스 수: {len(model.names)}")
print(f"클래스 목록 (일부): {list(model.names.values())[:10]}...")

# ==============================================
# 2단계: 이미지 탐지
# ==============================================
print("\n" + "=" * 60)
print("2. 이미지에서 객체 탐지")
print("=" * 60)

# 테스트 이미지 생성 (실제로는 자기 이미지 사용)
# results = model("path/to/your/image.jpg")

# 예시: URL에서 이미지 탐지
results = model("https://ultralytics.com/images/bus.jpg")

for r in results:
    print(f"\n탐지된 객체:")
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  {cls_name}: {confidence:.2f} ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

    # 결과 이미지 저장
    r.save(filename="detection_result.jpg")
    print(f"\n결과 이미지 저장: detection_result.jpg")

# ==============================================
# 3단계: 웹캠 실시간 탐지
# ==============================================
print("\n" + "=" * 60)
print("3. 웹캠 실시간 객체 탐지")
print("=" * 60)
print("'q' 키를 누르면 종료됩니다.")

def run_webcam_detection():
    """웹캠으로 실시간 객체 탐지"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 추론
        results = model(frame, verbose=False)

        # 결과 그리기
        annotated = results[0].plot()

        # FPS 표시
        cv2.imshow("YOLOv8 Real-time Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 웹캠 실행 (주석 해제하여 사용)
# run_webcam_detection()

# ==============================================
# 4단계: 비디오 파일 탐지
# ==============================================
print("\n" + "=" * 60)
print("4. 비디오 파일 탐지")
print("=" * 60)

def process_video(video_path, output_path="output_video.mp4"):
    """비디오 파일에서 객체 탐지"""
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"  처리 중: {frame_count} 프레임...")

    cap.release()
    writer.release()
    print(f"  완료! {frame_count} 프레임 처리 → {output_path}")

# 사용법:
# process_video("my_video.mp4")

# ==============================================
# ★ 과제 ★
# ==============================================
"""
[실습 과제]
1. 웹캠 실시간 탐지를 실행하고, 탐지되는 객체를 확인하세요.
2. yolov8n, yolov8s, yolov8m 모델의 속도와 정확도를 비교하세요.
3. 특정 클래스만 탐지하도록 수정하세요 (예: 사람만).
   힌트: results[0].boxes에서 cls로 필터링
4. 탐지 결과를 CSV로 저장하는 기능을 추가하세요.
5. 다음 파일(02)에서 커스텀 데이터셋 학습을 해보세요!
"""
