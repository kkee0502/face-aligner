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

st.set_page_config(page_title="Precision Face Aligner", layout="wide")
st.title("📸 AI 얼굴 정밀 동기화 정렬기")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

def align_precision(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # 1. 기준점 추출
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    nose_bridge = np.array([landmarks[6].x * w, landmarks[6].y * h]) # 미간
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])    # 턱끝
    
    # 2. 회전 각도 계산 (눈 수평 유지)
    dY = r_eye[1] - l_eye[1]
    dX = r_eye[0] - l_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # 3. 배율 결정 및 측면 보정
    face_height_pixel = np.sqrt((nose_bridge[0] - chin[0])**2 + (nose_bridge[1] - chin[1])**2)
    
    # 정면/측면 판별 계수 (눈 너비와 얼굴 높이 비율)
    eye_dist = np.sqrt(dX**2 + dY**2)
    side_factor = eye_dist / face_height_pixel 
    
    # [핵심 보정] 측면(side_factor < 0.5)일수록 배율을 더 많이 깎아서 정면과 크기를 맞춤
    # 정면은 1.0, 측면일수록 0.82까지 배율을 줄임
    profile_compensation = 1.0 if side_factor > 0.55 else 0.82
    
    target_face_height = h * 0.30 # 전체 화면의 30%를 얼굴 높이로 설정
    scale = (target_face_height / face_height_pixel) * profile_compensation
    
    # 4. 변환 행렬 생성
    M = cv2.getRotationMatrix2D(tuple(nose_bridge), angle, scale)
    
    # [5. 턱 높이 강제 일치 로직]
    # 모든 사진에서 턱(Chin)의 Y축 위치를 화면 상단에서 65% 지점으로 고정
    # 정면/측면 모두 동일한 수평선상에 턱이 오게 됩니다.
    target_chin_y = h * 0.65
    target_chin_x = w * 0.5
    
    # 현재 턱 위치가 변환 후 어디로 가는지 계산
    curr_chin_transformed = M @ np.array([chin[0], chin[1], 1])
    
    # 목표 턱 위치와의 오차만큼 전체 이미지를 수직/수평 이동
    M[0, 2] += (target_chin_x - curr_chin_transformed[0])
    M[1, 2] += (target_chin_y - curr_chin_transformed[1])
    
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return aligned_img

uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(3)
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_precision(img_array)
        
        with cols[idx % 3]:
            if result is not None:
                st.image(result, caption=f"정렬 완료: {uploaded_file.name}", use_column_width=True)
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button(label="💾 다운로드", data=buf.getvalue(), file_name=f"aligned_{uploaded_file.name}", mime="image/png", key=f"dl_{idx}")
            else:
                st.warning(f"{uploaded_file.name}: 인식 실패")
