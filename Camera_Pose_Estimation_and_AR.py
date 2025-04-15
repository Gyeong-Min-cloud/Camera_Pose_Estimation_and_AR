import cv2 as cv
import numpy as np
import os
import time

# === 체스보드 설정 ===
board_pattern = (8, 6)  # 내부 코너 수
board_cellsize = 25  # mm 단위

# === 3D 객체 포인트 ===
obj_points_single = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
obj_points_single[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
obj_points_single *= board_cellsize

# === 포인트 리스트 ===
obj_points_list = []
img_points_list = []

# === 디버그 폴더 생성 ===
os.makedirs("debug_frames", exist_ok=True)

# === 비디오 로드 ===
video_file = "data/chessboard.mp4"
cap = cv.VideoCapture(video_file)

frame_id = 0
last_gray = None
print("▶️ 체스보드 코너 감지 및 캘리브레이션 시작...")

# === 체스보드 코너 탐색 루프 ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv.findChessboardCorners(gray, board_pattern, flags)

    if found and corners.shape[0] == board_pattern[0] * board_pattern[1]:
        corners2 = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        )
        obj_points_list.append(obj_points_single)
        img_points_list.append(corners2)
        last_gray = gray.copy()
    else:
        print(f"[WARN] Frame {frame_id}: 코너 감지 실패")
        cv.imwrite(f"debug_frames/fail_frame_{frame_id}.jpg", frame)

    frame_id += 1

cap.release()

# === 보정 수행 ===
if last_gray is None or len(obj_points_list) < 5:
    print("❌ 보정 실패: 감지된 체스보드 수가 부족합니다.")
    exit()

max_samples = min(len(obj_points_list), 50)  # 샘플 제한
obj_sample = obj_points_list[:max_samples]
img_sample = img_points_list[:max_samples]

print(f"📌 calibrateCamera() 호출 - 샘플 {len(obj_sample)}개 사용")
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1e-5)

start_time = time.time()
ret, K, dist_coeff, rvecs, tvecs = cv.calibrateCamera(
    obj_sample, img_sample, last_gray.shape[::-1],
    None, None, criteria=criteria
)
elapsed_time = time.time() - start_time

print(f"✅ 보정 완료! ⏱️ {elapsed_time:.2f}초")
print(f"fx: {K[0,0]:.4f}, fy: {K[1,1]:.4f}, cx: {K[0,2]:.4f}, cy: {K[1,2]:.4f}")
print(f"왜곡 계수: {dist_coeff.ravel()}")

# === 사각뿔 모양 정의 ===
# 바닥 사각형 4점 + 꼭짓점 1점
pyramid_base = (board_cellsize * np.array([
    [4, 2, 0], [5, 2, 0], [5, 4, 0], [4, 4, 0]
], dtype=np.float32))

pyramid_tip = (board_cellsize * np.array([[4.5, 3, -1.5]], dtype=np.float32))  # 꼭짓점

# === AR 시각화 루프 ===
cap = cv.VideoCapture(video_file)
print("🕶️ AR 시각화 시작...")

while cap.isOpened():
    valid, img = cap.read()
    if not valid:
        break

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, board_pattern, None)

    if found and corners.shape[0] == board_pattern[0] * board_pattern[1]:
        corners2 = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        )

        ret, rvec, tvec = cv.solvePnP(
            obj_points_single, corners2, K, dist_coeff
        )

        img_base, _ = cv.projectPoints(pyramid_base, rvec, tvec, K, dist_coeff)
        img_tip, _ = cv.projectPoints(pyramid_tip, rvec, tvec, K, dist_coeff)

        img_base = np.int32(img_base).reshape(-1, 2)
        tip = tuple(np.int32(img_tip[0].ravel()))

        # 바닥면 그리기
        cv.polylines(img, [img_base], True, (255, 0, 0), 2)

        # 바닥면 각 점에서 꼭짓점으로 선 그리기
        for point in img_base:
            cv.line(img, tuple(point), tip, (0, 255, 0), 2)

        # 카메라 위치 출력
        R, _ = cv.Rodrigues(rvec)
        cam_position = (-R.T @ tvec).flatten()
        info = f"XYZ: [{cam_position[0]:.2f}, {cam_position[1]:.2f}, {cam_position[2]:.2f}]"
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.imshow("AR Chessboard - Pyramid", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
print("🔚 AR 시각화 종료.")
