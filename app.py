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

st.set_page_config(page_title="Pixel-Line Aligner", layout="wide")
st.title("📸 전각도 라인 동기화 정렬기")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

def align_ultimate_sync(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # [1] 정밀 기준점 추출
    l_eye_inner = np.array([landmarks[133].x * w, landmarks[133].y * h]) # 왼쪽 눈 안쪽
    r_eye_inner = np.array([landmarks[362].x * w, landmarks[362].y * h]) # 오른쪽 눈 안쪽
    nose_bridge = np.array([landmarks[6].x * w, landmarks[6].y * h])    # 미간
    nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])       # 코끝
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])       # 턱끝
    
    # [2] 회전 각도 계산 (눈 수평 유지)
    dY = r_eye_inner[1] - l_eye_inner[1]
    dX = r_eye_inner[0] - l_eye_inner[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # [3] 배율 결정 (가장 중요한 부분)
    # 측면일 때도 변하지 않는 '미간~코끝'의 수직 투영 길이를 기준으로 삼습니다.
    # 기존 '미간~턱'보다 '미간~코끝'이 측면 회전 시 오차가 훨씬 적습니다.
    vert_dist = np.sqrt((nose_bridge[0] - nose_tip[0])**2 + (nose_bridge[1] - nose_tip[1])**2)
    
    # 측면 판별 (눈 사이 거리 대비 코 높이 비율)
    eye_dist = np.sqrt(dX**2 + dY**2)
    is_profile = (eye_dist / vert_dist) < 2.5 # 측면일 때 true
    
    # 배율 설정: 정면일 때의 기준을 잡고, 측면은 수치적으로 더 축소(0.75)하여 시각적 면적을 맞춤
    target_vert_dist = h * 0.08 # 코 높이 기준 배율
    profile_scale_fix = 0.75 if is_profile else 1.0 
    scale = (target_vert_dist / vert_dist) * profile_scale_fix
    
    # [4] 변환 행렬 생성 (유사 변환)
    M = cv2.getRotationMatrix2D(tuple(nose_bridge), angle, scale)
    
    # [5] 빨간 선(가이드라인)에 맞추기 위한 위치 보정
    # 1. 눈썹/눈 라인: 미간(nose_bridge)을 화면 상단 42% 지점에 고정
    # 2. 턱 라인: 턱(chin)을 화면 상단 65% 지점에 고정하도록 수직 이동량 미세 조정
    
    target_bridge_y = h * 0.42
    target_bridge_x = w * 0.5
    
    # 변환 후 미간 위치 계산
    curr_bridge_trans = M @ np.array([nose_bridge[0], nose_bridge[1], 1])
    
    M[0, 2] += (target_bridge_x - curr_bridge_trans[0])
    M[1, 2] += (target_bridge_y - curr_bridge_trans[1])
    
    # [6] 측면 사진 턱/눈썹 라인 최종 보정 (Offset)
    if is_profile:
        # 측면에서 턱이 내려가는 현상을 막기 위해 이미지를 위로 더 끌어올림
        M[1, 2] -= (h * 0.045) # 4.5% 추가 인상
    
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return aligned_img

# [7] UI 로직
uploaded_files = st.file_uploader("사진 세트를 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(len(uploaded_files) if len(uploaded_files) > 0 else 1)
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_ultimate_sync(img_array)
        
        with cols[idx]:
            if result is not None:
                st.image(result, caption=f"정렬됨: {uploaded_file.name}", use_column_width=True)
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾 다운로드", buf.getvalue(), f"final_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
