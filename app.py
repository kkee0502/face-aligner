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

st.set_page_config(page_title="Personal Set Aligner", layout="wide")
st.title("📸 세트별 라인 동기화 정렬기")
st.write("각 인물의 정면과 측면 사진에서 눈과 턱의 수평선을 일치시킵니다.")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

def align_set_perfect(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # 1. 핵심 랜드마크 추출
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    nose_bridge = np.array([landmarks[6].x * w, landmarks[6].y * h]) # 미간
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])    # 턱끝
    
    # 2. 얼굴 각도 및 상태 분석
    dY = r_eye[1] - l_eye[1]
    dX = r_eye[0] - l_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # 눈 사이 거리와 수직 높이 측정
    eye_dist = np.sqrt(dX**2 + dY**2)
    v_height = np.sqrt((nose_bridge[0] - chin[0])**2 + (nose_bridge[1] - chin[1])**2)
    
    # [핵심] 측면도(Profile-ness) 계산
    # 정면은 보통 0.6 이상, 측면은 0.4 이하로 떨어집니다.
    side_score = eye_dist / v_height
    is_profile = side_score < 0.52
    
    # 3. 배율 보정 (측면 사진이 커지는 현상 방지)
    # 얼굴이 돌아가면 수직 거리(미간-턱)가 미세하게 짧게 측정되는 것을 보정
    # 보정 계수를 0.82로 적용하여 정면 면적과 시각적으로 일치시킵니다.
    profile_scale_adj = 0.82 if is_profile else 1.0
    
    target_v_height = h * 0.32
    scale = (target_v_height / v_height) * profile_scale_adj
    
    # 4. 변환 행렬 생성
    M = cv2.getRotationMatrix2D(tuple(nose_bridge), angle, scale)
    
    # [5. 라인 동기화의 핵심: 수직 오프셋 보정]
    # 정면에서는 턱이 낮게 잡히고, 측면에서는 고개 각도에 따라 턱 위치가 변합니다.
    # 모든 사진의 '미간' 높이를 40% 지점에 고정하면 눈 높이가 맞습니다.
    target_bridge_y = h * 0.40
    target_bridge_x = w * 0.5
    
    # 변환 후 미간의 위치 계산
    curr_bridge_trans = M @ np.array([nose_bridge[0], nose_bridge[1], 1])
    
    M[0, 2] += (target_bridge_x - curr_bridge_trans[0])
    M[1, 2] += (target_bridge_y - curr_bridge_trans[1])
    
    # [6. 측면 사진 전용 턱 들기 보정]
    # 측면 사진에서 턱이 정면보다 아래로 쳐지는 현상을 막기 위해
    # 이미지 자체를 위로 살짝 더 밀어 올립니다 (전체 높이의 2%~3% 추가 상승)
    if is_profile:
        M[1, 2] -= (h * 0.032) # 이 수치가 높을수록 측면 사진의 턱이 위로 올라갑니다.

    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return aligned_img

uploaded_files = st.file_uploader("인물 세트 사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(3)
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_set_perfect(img_array)
        
        with cols[idx % 3]:
            if result is not None:
                st.image(result, caption=f"정렬됨: {uploaded_file.name}", use_column_width=True)
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾 다운로드", buf.getvalue(), f"aligned_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
