import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import sys
import os

# [핵심] 라이브러리 경로 강제 검색 및 로드
def load_ai_engine():
    try:
        import mediapipe as mp
        # 가끔 mp.solutions가 안 보일 때를 대비해 하위 모듈 강제 로드
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
        return mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1, 
            refine_landmarks=True
        )
    except ImportError:
        # 설치는 되었으나 경로 인식이 안 될 때 sys.path를 뒤집니다.
        import site
        sys.path.append(site.getsitepackages()[0])
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# 페이지 설정
st.set_page_config(page_title="Face Aligner", layout="centered")
st.title("📸 AI 얼굴 각도 정렬기")

# 엔진 초기화
if 'engine' not in st.session_state:
    try:
        st.session_state.engine = load_ai_engine()
        st.success("✅ AI 엔진이 정상적으로 로드되었습니다.")
    except Exception as e:
        st.error(f"❌ AI 로드 실패: {e}")
        st.stop()

face_mesh = st.session_state.engine

# 사진 업로드 및 처리 로직 (이하는 이전과 동일하지만 더 견고하게 수정)
uploaded_files = st.file_uploader("사진을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        
        # AI 처리
        results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = img_array.shape
            
            # 눈/코 기준점 계산
            l_eye, r_eye, nose = landmarks[33], landmarks[263], landmarks[1]
            center_y = int((l_eye.y + r_eye.y) / 2 * h)
            center_x = int(nose.x * w)
            
            # 크롭 및 리사이즈
            dist = np.sqrt((l_eye.x - r_eye.x)**2 + (l_eye.y - r_eye.y)**2)
            sz = int((min(h, w) * 0.4) / (0.25 / dist))
            y1, y2 = max(0, center_y-sz), min(h, center_y+sz)
            x1, x2 = max(0, center_x-sz), min(w, center_x+sz)
            
            res = cv2.resize(img_array[y1:y2, x1:x2], (512, 512), interpolation=cv2.INTER_LANCZOS4)
            
            st.image(res, caption=f"정렬됨: {uploaded_file.name}")
            
            # 다운로드 버튼
            res_img = Image.fromarray(res)
            buf = io.BytesIO()
            res_img.save(buf, format="PNG")
            st.download_button("💾 다운로드", buf.getvalue(), f"aligned_{uploaded_file.name}", "image/png")
        else:
            st.warning(f"{uploaded_file.name}: 얼굴을 찾을 수 없습니다.")
