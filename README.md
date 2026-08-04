# 🎯 aim-code

손과 손에 든 물건을 실시간으로 인식·추적하는 락온 트래커

## 📌 소개
웹캠 영상에서 물체와 손을 동시에 감지하고, 게임 스타일의 락온 UI로
타겟을 조준·추적하는 컴퓨터 비전 프로젝트입니다. (개인 프로젝트, 전체 개발)

## ⚙️ 동작 방식
1. **YOLOv8**로 대상 물체(휴대폰, 물병) 감지 → 신뢰도가 가장 높은 타겟에 락온
2. 물체가 없으면 **MediaPipe Hands**로 손목 좌표를 추적
3. 둘 다 없으면 화면 중앙에서 스캔 대기
4. 조준점은 **지수 스무딩**으로 부드럽게 이동 (위치·크기 모두 보정)

## 🛠 기술 스택
`Python` `OpenCV` `YOLOv8 (Ultralytics)` `MediaPipe` `NumPy`

## ✨ 구현 포인트
- 물체 감지 + 손 인식 두 모델을 하나의 실시간 파이프라인으로 통합
- LOCKED → TRACKING → SCANNING 3단계 상태 전환 로직
- 코너 브래킷 + 크로스헤어 + 펄스 원(알파 블렌딩) 락온 UI를 OpenCV로 직접 렌더링

## 🚀 실행
​```bash
pip install opencv-python mediapipe ultralytics
python main.py   # q 키로 종료
​```
