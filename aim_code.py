import cv2
import mediapipe as mp
import numpy as np
import math
import time
from ultralytics import YOLO

# ✅ 추적할 YOLO 클래스 이름 (COCO 기준)
TARGET_OBJECTS = ['cell phone', 'bottle']

# YOLO 모델 로드
model = YOLO('yolov8n.pt')

# MediaPipe 손 인식
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ==================== 락온 UI ====================
def draw_lockon(img, cx, cy, size, label='', conf=0.0, color=(0, 255, 0)):
    half = size // 2
    corner = size // 4
    t = 2

    for x, y, dx, dy in [
        (cx-half, cy-half,  1,  1),
        (cx+half, cy-half, -1,  1),
        (cx-half, cy+half,  1, -1),
        (cx+half, cy+half, -1, -1),
    ]:
        cv2.line(img, (x, y), (x + corner*dx, y), color, t)
        cv2.line(img, (x, y), (x, y + corner*dy), color, t)

    cv2.line(img, (cx-12, cy), (cx+12, cy), color, 1)
    cv2.line(img, (cx, cy-12), (cx, cy+12), color, 1)

    radius = half
    alpha = abs(math.sin(time.time() * 4))
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), radius, color, 1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    if label:
        cv2.putText(img, f'{label.upper()}  {conf*100:.0f}%',
                    (cx - half, cy - half - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# ==================== 손 관절 그리기 ====================
def draw_hand(img, hand_landmarks):
    mp_drawing.draw_landmarks(
        img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2))

# ==================== 메인 ====================
cap = cv2.VideoCapture(0)
print("🎯 추적 시작! q를 누르면 종료")

aim_x, aim_y = 0, 0
aim_size = 100
smoothing = 0.8

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    h, w = img.shape[:2]

    if aim_x == 0:
        aim_x, aim_y = w//2, h//2

    # ========== YOLO 물체 감지 ==========
    results = model(img, verbose=False)[0]
    target_found = False
    best_box = None
    best_conf = 0
    best_label = ''

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        if label in TARGET_OBJECTS and conf > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 1)

            if conf > best_conf:
                best_conf = conf
                best_label = label
                best_box = (x1, y1, x2, y2)
                target_found = True

    # ========== MediaPipe 손 인식 ==========
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    hand_cx, hand_cy = None, None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            draw_hand(img, hand_landmarks)
            wrist = hand_landmarks.landmark[0]
            hand_cx = int(wrist.x * w)
            hand_cy = int(wrist.y * h)

    # ========== 에임 타겟 결정 ==========
    if target_found and best_box:
        x1, y1, x2, y2 = best_box
        target_cx = (x1 + x2) // 2
        target_cy = (y1 + y2) // 2
        target_size = max(x2-x1, y2-y1)
        color = (0, 255, 0)
        status = f'LOCKED: {best_label.upper()}'

    elif hand_cx and hand_cy:
        target_cx = hand_cx
        target_cy = hand_cy
        target_size = 120
        color = (0, 200, 255)
        status = 'TRACKING: HAND'

    else:
        target_cx = w // 2
        target_cy = h // 2
        target_size = 100
        color = (80, 80, 80)
        status = 'SCANNING...'

    # ========== 에임 부드럽게 이동 ==========
    aim_x = int(aim_x + (target_cx - aim_x) * smoothing)
    aim_y = int(aim_y + (target_cy - aim_y) * smoothing)
    aim_size = int(aim_size + (target_size - aim_size) * smoothing)

    # ========== 락온 UI 그리기 ==========
    draw_lockon(img, aim_x, aim_y, aim_size,
                label=best_label if target_found else ('HAND' if hand_cx else ''),
                conf=best_conf if target_found else 1.0,
                color=color)

    cv2.putText(img, status, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(img, 'q: quit', (w-90, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

    cv2.imshow('LOCK-ON TRACKER', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()