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

st.set_page_config(page_title="Line-Lock Aligner", layout="wide")
st.title("📸 정밀 라인 고정 정렬기")
st.write("정면과 측면의 턱선, 눈썹 높이를 수학적으로 완벽히 일치시킵니다.")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

def align_precise_line_lock(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # [1] 정밀 포인트 추출 (수직 정렬의 핵심)
    # 미간(6번)과 턱 끝(152번)을 사용하여 얼굴의 '진짜 수직 길이'를 측정합니다.
    bridge = np.array([landmarks[6].x * w, landmarks[6].y * h])
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    
    # 수평 각도 계산 (두 눈: 33번, 263번)
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

    # [2] 배율 설정 (핵심 수정 사항)
    # 측면/정면 구분 없이 '미간~턱' 거리가 전체 높이의 30%가 되도록 스케일을 잡습니다.
    # 이렇게 하면 측면 사진이 과하게 커지는 현상이 원천 차단됩니다.
    current_face_height = np.linalg.norm(bridge - chin)
    target_face_height = h * 0.30 
    scale = target_face_height / current_face_height
    
    # [3] 변환 행렬 생성 (미간 중심 회전 및 스케일)
    M = cv2.getRotationMatrix2D(tuple(bridge), angle, scale)
    
    # [4] 라인 고정 로직 (Line-Lock)
    # 변환된 미간 위치가 어디인지 확인
    curr_bridge_trans = M @ np.array([bridge[0], bridge[1], 1])
    
    # 모든 사진의 미간을 가로 50%, 세로 40% 지점으로 '못박기'
    M[0, 2] += (w * 0.5 - curr_bridge_trans[0])
    M[1, 2] += (h * 0.40 - curr_bridge_trans[1])
    
    # [5] 이미지 생성
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return aligned_img

# --- UI 레이아웃 유지 ---
uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(len(uploaded_files))
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_precise_line_lock(img_array)
        
        with cols[idx]:
            if result is not None:
                st.image(result, caption=f"라인 동기화: {uploaded_file.name}", use_column_width=True)
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾 다운로드", buf.getvalue(), f"locked_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
