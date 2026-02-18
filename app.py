import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def load_ai_engine():
    try:
        import mediapipe as mp
        from mediapipe.solutions import face_mesh as mp_face_mesh
        return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    except:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

st.set_page_config(page_title="No-Distortion Aligner", layout="wide")
st.title("📸 왜곡 없는 얼굴 정렬기")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

def align_face_no_distortion(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # [1] 두 눈의 좌표 추출
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    
    # [2] 회전 각도 및 눈 사이 거리 계산
    dY = r_eye[1] - l_eye[1]
    dX = r_eye[0] - l_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # 현재 눈 사이 거리
    current_dist = np.sqrt(dX**2 + dY**2)
    
    # [3] 목표 설정
    # 모든 사진의 눈 사이 거리를 화면 짧은 쪽의 30%로 통일 (얼굴 크기 고정)
    target_dist = min(h, w) * 0.30
    scale = target_dist / current_dist
    
    # [4] 유사 변환 행렬 생성 (회전 + 배율 + 이동)
    # 이미지의 형태를 왜곡하지 않고 회전과 크기만 조절합니다.
    eyes_center = ((l_eye[0] + r_eye[0]) // 2, (l_eye[1] + r_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    
    # [5] 눈 위치를 사진의 특정 지점(중앙 상단)으로 이동시키기 위한 보정
    tX = w * 0.5
    tY = h * 0.45
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])
    
    # 최종 변환
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return aligned_img

uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(3)
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_face_no_distortion(img_array)
        
        with cols[idx % 3]:
            if result is not None:
                st.image(result, caption=f"정렬 완료: {uploaded_file.name}", use_column_width=True)
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾 다운로드", buf.getvalue(), f"aligned_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
