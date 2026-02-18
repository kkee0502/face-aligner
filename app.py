import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# [1. AI 엔진 로드]
def load_ai_engine():
    try:
        import mediapipe as mp
        from mediapipe.solutions import face_mesh as mp_face_mesh
        return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    except:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

st.set_page_config(page_title="Universal Face Aligner", layout="wide")
st.title("📸 AI 전각도 얼굴 정렬기 (측면 크기 보정)")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()

face_mesh = st.session_state.engine

# [2. 핵심 정렬 함수]
def align_face_universal(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # 기준점 추출
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    nose_bridge = np.array([landmarks[6].x * w, landmarks[6].y * h])
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    
    # 1. 회전 각도 계산 (눈 수평 유지)
    dY = r_eye[1] - l_eye[1]
    dX = r_eye[0] - l_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # 2. 배율 계산 (측면 정밀 보정 로직)
    # 수직 길이 측정 (미간 ~ 턱)
    face_height_pixel = np.sqrt((nose_bridge[0] - chin[0])**2 + (nose_bridge[1] - chin[1])**2)
    
    # [측면 보정 계수 계산]
    # 정면일수록 눈 사이 거리(eye_width)가 길고, 측면일수록 짧아집니다.
    eye_width = np.sqrt(dX**2 + dY**2)
    # 얼굴 높이 대비 눈 너비의 비율을 구함 (정면은 보통 0.6~0.7, 측면은 0.3 이하로 떨어짐)
    aspect_ratio = eye_width / face_height_pixel
    
    # 측면 보정: 측면(aspect_ratio가 작음)일수록 scale을 미세하게 낮춤 (0.9 ~ 1.0 사이 조절)
    # 얼굴이 많이 돌아갔을 때(측면) 사진이 커지는 것을 방지하기 위해 0.92 정도의 상수를 곱해줍니다.
    profile_compensation = 1.0 if aspect_ratio > 0.5 else 0.92
    
    target_face_height = h * 0.35 
    scale = (target_face_height / face_height_pixel) * profile_compensation
    
    # 3. 유사 변환 행렬 생성 (왜곡 방지)
    M = cv2.getRotationMatrix2D(tuple(nose_bridge), angle, scale)
    
    # 4. 위치 고정 (중앙 50%, 상단 42%)
    tX = w * 0.5
    tY = h * 0.42
    M[0, 2] += (tX - nose_bridge[0])
    M[1, 2] += (tY - nose_bridge[1])
    
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return aligned_img

# [3. UI 부분]
uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(3)
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_face_universal(img_array)
        
        with cols[idx % 3]:
            if result is not None:
                st.image(result, caption=f"보정완료: {uploaded_file.name}", use_column_width=True)
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button(label="💾 다운로드", data=buf.getvalue(), file_name=f"aligned_{uploaded_file.name}", mime="image/png", key=f"dl_{idx}")
            else:
                st.warning(f"{uploaded_file.name}: 인식 실패")
