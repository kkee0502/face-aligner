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

st.set_page_config(page_title="Final Chin-Line Sync", layout="wide")
st.title("📸 전각도 턱 라인 일치 정렬기")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

def align_face_fixed_line(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # 1. 정밀 기준점 추출
    l_eye_inner = np.array([landmarks[133].x * w, landmarks[133].y * h])
    r_eye_inner = np.array([landmarks[362].x * w, landmarks[362].y * h])
    nose_bridge = np.array([landmarks[6].x * w, landmarks[6].y * h]) # 미간
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])    # 턱끝
    
    # 2. 회전 각도 계산 (눈 수평 유지)
    dY = r_eye_inner[1] - l_eye_inner[1]
    dX = r_eye_inner[0] - l_eye_inner[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # 3. 배율 결정 (수직 기둥 기준)
    face_height_pixel = np.sqrt((nose_bridge[0] - chin[0])**2 + (nose_bridge[1] - chin[1])**2)
    
    # 측면 판별 (눈 너비 비율)
    eye_dist = np.sqrt(dX**2 + dY**2)
    is_profile = (eye_dist / face_height_pixel) < 0.52
    
    # [수정] 측면 배율을 더 과감하게 축소 (0.72) 하여 정면과 면적을 맞춤
    profile_scale_fix = 0.72 if is_profile else 1.0
    target_face_height = h * 0.28  # 얼굴 크기 표준화
    scale = (target_face_height / face_height_pixel) * profile_scale_fix
    
    # 4. 변환 행렬 생성
    M = cv2.getRotationMatrix2D(tuple(nose_bridge), angle, scale)
    
    # [5. 턱 라인 강제 일치 로직]
    # 모든 사진의 턱(Chin) 끝이 화면 상단에서 정확히 68% 지점에 오도록 설정
    # 측면일 때 턱이 더 내려오는 현상을 막기 위해 target_y를 인위적으로 상향 조정
    target_chin_y = h * 0.64 if is_profile else h * 0.68
    target_chin_x = w * 0.5
    
    # 변환 후의 현재 턱 위치 계산
    curr_chin_trans = M @ np.array([chin[0], chin[1], 1])
    
    # 턱의 오차만큼 전체 이미지를 수직/수평 이동
    M[0, 2] += (target_chin_x - curr_chin_trans[0])
    M[1, 2] += (target_chin_y - curr_chin_trans[1])
    
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return aligned_img

uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(len(uploaded_files))
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_face_fixed_line(img_array)
        
        with cols[idx]:
            if result is not None:
                st.image(result, caption=f"정렬됨: {uploaded_file.name}", use_column_width=True)
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾 다운로드", buf.getvalue(), f"final_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
