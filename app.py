import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# [수정 포인트] Mediapipe를 가장 안전하게 불러오는 방식
try:
    import mediapipe as mp
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True
    )
except Exception as e:
    st.error(f"AI 엔진 로드 중 오류가 발생했습니다: {e}")

st.set_page_config(page_title="Face Aligner", layout="wide")
st.title("📸 AI 얼굴 각도 정렬기")
st.write("사진을 올리면 눈 높이와 얼굴 크기를 자동으로 맞춰줍니다.")

uploaded_files = st.file_uploader("사진들을 업로드하세요 (여러 장 가능)", accept_multiple_files=True)

def process_image(img_array):
    h, w, _ = img_array.shape
    # BGR 변환 후 처리
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    # 기준점 추출
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose_tip = landmarks[1]

    # 1. 눈 높이 중심 계산 (Y축)
    center_y = int((left_eye.y + right_eye.y) / 2 * h)
    
    # 2. 정면/측면에 따른 중심 계산 (X축)
    is_profile = abs(left_eye.z - right_eye.z) > 0.1
    center_x = int((left_eye.x + right_eye.x) / 2 * w) if is_profile else int(nose_tip.x * w)

    # 3. 배율 조정 (눈 사이 거리 기준)
    eye_dist = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)
    scale = 0.25 / eye_dist if eye_dist > 0 else 1.0
    
    # 4. 크롭 영역 설정
    crop_size = int((min(h, w) * 0.4) / scale)
    y1, y2 = max(0, center_y - crop_size), min(h, center_y + crop_size)
    x1, x2 = max(0, center_x - crop_size), min(w, center_x + crop_size)
    
    cropped = img_array[y1:y2, x1:x2]
    # 결과가 너무 작을 경우를 대비해 512x512로 통일
    return cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_LANCZOS4)

if uploaded_files:
    # 사진들을 가로로 배치
    cols = st.columns(len(uploaded_files))
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = process_image(img_array)
        
        with cols[idx]:
            if result is not None:
                st.image(result, caption=uploaded_file.name, use_container_width=True)
                # 다운로드 버튼
                result_img = Image.fromarray(result)
                buf = io.BytesIO()
                result_img.save(buf, format
