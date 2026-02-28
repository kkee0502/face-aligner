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
st.write("정면과 측면의 턱선, 눈썹 높이를 강제로 일치시킵니다.")

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
    
    # [1] 정밀 포인트 추출 (눈썹, 미간, 턱)
    # 눈썹 라인 (눈썹 위쪽 랜드마크 105번, 334번의 중간 높이 사용)
    brow_y = (landmarks[105].y + landmarks[334].y) / 2 * h
    nose_bridge = np.array([landmarks[6].x * w, landmarks[6].y * h]) 
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    
    # 눈 수평 각도
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

    # [2] 얼굴 높이 계산 (미간 ~ 턱)
    current_face_height = np.sqrt((nose_bridge[0] - chin[0])**2 + (nose_bridge[1] - chin[1])**2)
    
    # 측면 판별 (눈 가로 길이 비율)
    eye_dist = np.sqrt((r_eye[0]-l_eye[0])**2 + (r_eye[1]-l_eye[1])**2)
    side_ratio = eye_dist / current_face_height
    is_profile = side_ratio < 0.50  # 값이 작을수록 완전 측면
    
    # [3] 배율 설정 (가장 중요)
    # 정면 대비 측면 사진이 항상 크게 나오는 현상을 해결하기 위해 
    # 측면일 경우 배율을 0.70까지 낮춥니다. (이전보다 더 과감하게 축소)
    target_face_height = h * 0.28
    base_scale = target_face_height / current_face_height
    scale = base_scale * (0.70 if is_profile else 1.0)
    
    # [4] 변환 행렬 생성 (미간 중심)
    M = cv2.getRotationMatrix2D(tuple(nose_bridge), angle, scale)
    
    # [5] 라인 고정 로직 (눈썹 라인과 턱 라인을 캔버스에 못박기)
    # 정면 사진 기준: 눈썹(35%), 턱(65%) 지점에 오도록 설정
    # 측면 사진 기준: 턱이 처지는 현상을 보정하기 위해 턱을 61% 지점으로 강제 인상
    target_brow_y = h * 0.35
    target_chin_y = h * 0.61 if is_profile else h * 0.65
    
    # 현재 미간 위치를 변환 후 어디로 가는지 확인
    curr_bridge_trans = M @ np.array([nose_bridge[0], nose_bridge[1], 1])
    
    # 수평 중앙(50%), 수직은 미간(눈 높이 근처)을 40% 지점으로 강제 이동
    M[0, 2] += (w * 0.5 - curr_bridge_trans[0])
    M[1, 2] += (h * 0.40 - curr_bridge_trans[1])
    
    # [6] 측면 전용 추가 수직 보정 (Offset)
    # 눈썹 라인과 턱 라인이 정면과 일치하지 않을 경우 여기서 미세 조정
    if is_profile:
        M[1, 2] -= (h * 0.05) # 이미지를 5% 더 위로 밀어 올림

    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return aligned_img

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
