import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import sys

# [1. AI 엔진 로드 로직] - 경로 오류를 방지하기 위해 가장 안전한 방식을 사용합니다.
def load_ai_engine():
    try:
        import mediapipe as mp
        # 표준 경로 시도
        if hasattr(mp, 'solutions'):
            mp_face_mesh = mp.solutions.face_mesh
        else:
            from mediapipe.python.solutions import face_mesh as mp_face_mesh
        
        return mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1, 
            refine_landmarks=True
        )
    except Exception:
        # 최종 수단: 직접 모듈 경로 주입
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# 페이지 설정
st.set_page_config(page_title="Face Aligner Pro", layout="wide")
st.title("📸 AI 얼굴 각도 정렬기 (Pro)")
st.write("얼굴 여백을 넉넉히 확보하고 원본 사진의 비율을 유지합니다.")

# 세션 상태에 엔진 저장 (매번 로드 방지)
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
    
    # AI 인식 (BGR 변환 필수)
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # 중심점 잡기 (눈과 코 위치 기준)
    l_eye, r_eye, nose = landmarks[33], landmarks[263], landmarks[1]
    center_y = int((l_eye.y + r_eye.y) / 2 * h)
    center_x = int(nose.x * w)
    
    # 얼굴 크기 측정 (눈 사이 거리 기준)
    eye_dist = np.sqrt((l_eye.x - r_eye.x)**2 + (l_eye.y - r_eye.y)**2)
    
    # [수정포인트] 여백 설정 (숫자가 클수록 더 멀리서 찍은 것처럼 여백이 생깁니다)
    # 기존 0.4에서 0.8로 늘려 여백을 확보했습니다.
    zoom_factor = 0.8 / (0.25 / eye_dist)
    
    # 원본 비율을 유지하기 위해 가로/세로 잘라낼 폭 계산
    crop_w = int(w * zoom_factor)
    crop_h = int(h * zoom_factor)
    
    # 좌표 계산 (이미지 범위를 벗어나지 않게 처리)
    y1, y2 = max(0, center_y - crop_h // 2), min(h, center_y + crop_h // 2)
    x1, x2 = max(0, center_x - crop_w // 2), min(w, center_x + crop_w // 2)
    
    # 잘라내기 (Resize를 빼서 원본 비율을 유지)
    cropped = img_array[y1:y2, x1:x2]
    return cropped

# [3. 웹 화면 UI]
uploaded_files = st.file_uploader("사진들을 업로드하세요 (여러 장 가능)", accept_multiple_files=True)

if uploaded_files:
    # 사진 개수에 따라 동적으로 열 생성
    num_files = len(uploaded_files)
    cols = st.columns(min(num_files, 3)) # 한 줄에 최대 3장씩 표시
    
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        
        # 처리 실행
        result = process_face_keep_ratio(img_array)
        
        with cols[idx % 3]:
            if result is not None:
                st.image(result, caption=f"정렬됨: {uploaded_file.name}", use_container_width=True)
                
                # 다운로드 버튼 구현
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
                st.warning(f"{uploaded_file.name}: 얼굴 인식 실패")
