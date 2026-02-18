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

st.set_page_config(page_title="Chin-Aligned Face Fixer", layout="wide")
st.title("📸 AI 얼굴 크기 정렬기 (턱 위치 고정형)")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

def align_face_by_chin(img_array):
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
    
    # 2. 수평 회전 각도 계산
    dY = r_eye[1] - l_eye[1]
    dX = r_eye[0] - l_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # 3. 배율 결정 (수직 길이 기준)
    # 미간부터 턱까지의 길이를 측정합니다.
    face_height_pixel = np.sqrt((nose_bridge[0] - chin[0])**2 + (nose_bridge[1] - chin[1])**2)
    
    # 정면/측면 여부에 따른 미세 보정 (측면일 때 턱 거리가 짧게 측정되는 현상 보정)
    eye_dist = np.sqrt(dX**2 + dY**2)
    is_profile = (eye_dist / face_height_pixel) < 0.5
    
    # [수정 포인트] 측면일 때 배율을 더 공격적으로 낮춤 (0.88)
    profile_factor = 0.88 if is_profile else 1.0
    
    # 모든 사진의 얼굴 수직 길이를 화면 높이의 32%로 고정
    target_face_height = h * 0.32
    scale = (target_face_height / face_height_pixel) * profile_factor
    
    # 4. 변환 행렬 생성
    # 회전 중심은 미간으로 잡되, 이동의 기준은 '턱'으로 잡습니다.
    M = cv2.getRotationMatrix2D(tuple(nose_bridge), angle, scale)
    
    # [5. 턱 위치 고정]
    # 모든 사진에서 아래턱(chin)이 화면 가로 중앙 50%, 세로 70% 지점에 오도록 강제 고정
    target_chin_x = w * 0.5
    target_chin_y = h * 0.70
    
    # 현재 턱 위치를 변환 행렬에 대입하여 변환 후의 위치를 계산
    current_chin_transformed = M @ np.array([chin[0], chin[1], 1])
    
    # 목표 지점과의 차이만큼 이동량을 보정
    M[0, 2] += (target_chin_x - current_chin_transformed[0])
    M[1, 2] += (target_chin_y - current_chin_transformed[1])
    
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return aligned_img

uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(3)
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_face_by_chin(img_array)
        
        with cols[idx % 3]:
            if result is not None:
                st.image(result, caption=f"턱 위치 고정: {uploaded_file.name}", use_column_width=True)
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button(label="💾 다운로드", data=buf.getvalue(), file_name=f"aligned_{uploaded_file.name}", mime="image/png", key=f"dl_{idx}")
            else:
                st.warning(f"{uploaded_file.name}: 인식 실패")
