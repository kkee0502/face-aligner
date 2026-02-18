import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def load_ai_engine():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

st.set_page_config(page_title="Ultimate Frame & Face Sync", layout="wide")
st.title("📸 기계 프레임 & 안면 라인 완전 동기화")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

def align_image_final(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    
    # [1] 기계 고정 장치(하얀색 프레임 및 녹색 핀) 감지 로직
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    # 녹색 핀 인식 범위 정밀화
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # [2] 안면 랜드마크 추출
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    # --- Case A: 기계가 감지된 경우 (기계 기준 정렬) ---
    if len(contours) >= 1:
        # 가장 큰 컨투어(녹색 핀)의 중심점 계산
        c = max(contours, key=cv2.contourArea)
        M_cnt = cv2.moments(c)
        if M_cnt["m00"] != 0:
            cX = int(M_cnt["m10"] / M_cnt["m00"])
            cY = int(M_cnt["m01"] / M_cnt["m00"])
            
            # 기계 사진은 회전하지 않고(기계 자체가 수평이므로) 위치만 고정
            # 기계 고정핀 위치를 화면의 (75%, 50%) 지점으로 고정
            target_x, target_y = w * 0.75, h * 0.50
            M = np.float32([[1, 0, target_x - cX], [0, 1, target_y - cY]])
            return cv2.warpAffine(img_array, M, (w, h))

    # --- Case B: 기계가 없거나 인식이 안 된 경우 (얼굴 기준 정렬) ---
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 기준점 (눈 안쪽, 미간, 턱)
        l_eye = np.array([landmarks[133].x * w, landmarks[133].y * h])
        r_eye = np.array([landmarks[362].x * w, landmarks[362].y * h])
        bridge = np.array([landmarks[6].x * w, landmarks[6].y * h])
        chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
        
        # 1. 각도: 눈 수평
        angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))
        
        # 2. 배율: 측면 얼굴이 커지는 것을 막기 위해 수직 거리를 강제 고정
        curr_h = np.linalg.norm(bridge - chin)
        eye_dist = np.linalg.norm(r_eye - l_eye)
        is_profile = (eye_dist / curr_h) < 0.55
        
        # 측면일 때 배율을 25% 더 축소하여 정면 면적과 맞춤
        target_h = h * 0.30
        scale = (target_h / curr_h) * (0.75 if is_profile else 1.0)
        
        M = cv2.getRotationMatrix2D(tuple(bridge), angle, scale)
        
        # 3. 위치: 턱과 눈썹 라인 동기화
        # 턱 위치를 화면 70% 지점에 강제 고정 (측면은 기하학적 보정으로 67%에 배치)
        t_chin_y = h * 0.67 if is_profile else h * 0.70
        curr_chin_trans = M @ np.array([chin[0], chin[1], 1])
        
        M[0, 2] += (w * 0.5 - curr_chin_trans[0])
        M[1, 2] += (t_chin_y - curr_chin_trans[1])
        
        return cv2.warpAffine(img_array, M, (w, h))

    return img_array

uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(3)
    for idx, f in enumerate(uploaded_files):
        img = Image.open(f)
        img_arr = np.array(img.convert('RGB'))
        res = align_image_final(img_arr)
        with cols[idx % 3]:
            st.image(res, caption=f"정렬됨: {f.name}")
            buf = io.BytesIO(); Image.fromarray(res).save(buf, format="PNG")
            st.download_button("💾", buf.getvalue(), f"res_{f.name}", "image/png", key=idx)
