import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# [1. AI 엔진 로드 로직]
def load_ai_engine():
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions'):
            mp_face_mesh = mp.solutions.face_mesh
        else:
            from mediapipe.python.solutions import face_mesh as mp_face_mesh
        return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    except Exception:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# 페이지 설정
st.set_page_config(page_title="Face Aligner Pro", layout="wide")
st.title("📸 AI 얼굴 각도 정렬기 (Pro)")

if 'engine' not in st.session_state:
    try:
        st.session_state.engine = load_ai_engine()
    except Exception as e:
        st.error(f"AI 엔진 로드 실패: {e}")
        st.stop()

face_mesh = st.session_state.engine

# [2. 이미지 처리 함수]
def process_face_keep_ratio(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    l_eye, r_eye, nose = landmarks[33], landmarks[263], landmarks[1]
    
    # 중심점 잡기
    center_y = int((l_eye.y + r_eye.y) / 2 * h)
    center_x = int(nose.x * w)
    
    # 얼굴 크기 측정 (눈 사이 거리 기준)
    eye_dist = np.sqrt((l_eye.x - r_eye.x)**2 + (l_eye.y - r_eye.y)**2)
    
    # [수정포인트] 여백 수치 조절
    # 0.8: 시원한 여백 / 1.0: 아주 넓은 여백 / 0.5: 얼굴 위주
    zoom_factor = 0.8 / (0.25 / eye_dist)
    
    crop_w = int(w * zoom_factor)
    crop_h = int(h * zoom_factor)
    
    # 좌표 계산
    y1, y2 = max(0, center_y - crop_h // 2), min(h, center_y + crop_h // 2)
    x1, x2 = max(0, center_x - crop_w // 2), min(w, center_x + crop_w // 2)
    
    return img_array[y1:y2, x1:x2]

# [3. 웹 화면 UI]
uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    num_files = len(uploaded_files)
    cols = st.columns(min(num_files, 3))
    
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = process_face_keep_ratio(img_array)
        
        with cols[idx % 3]:
            if result is not None:
                # [오류 해결] use_container_width 대신 use_column_width=True 사용
                st.image(result, caption=f"정렬됨: {uploaded_file.name}", use_column_width=True)
                
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button(
                    label="💾 다운로드",
                    data=buf.getvalue(),
                    file_name=f"aligned_{uploaded_file.name}",
                    mime="image/png",
                    key=f"btn_{idx}"
                )
            else:
                st.warning(f"{uploaded_file.name}: 인식 실패")
