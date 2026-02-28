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
    
    # [1] 정밀 포인트 추출 (기존 코드 그대로 유지)
    brow_y = (landmarks[105].y + landmarks[334].y) / 2 * h
    nose_bridge = np.array([landmarks[6].x * w, landmarks[6].y * h]) 
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

    # [2] 얼굴 높이 계산 (기존 코드 그대로 유지)
    current_face_height = np.sqrt((nose_bridge[0] - chin[0])**2 + (nose_bridge[1] - chin[1])**2)
    
    eye_dist = np.sqrt((r_eye[0]-l_eye[0])**2 + (r_eye[1]-l_eye[1])**2)
    side_ratio = eye_dist / current_face_height
    is_profile = side_ratio < 0.50 
    
    # [3] 배율 설정 (기존 코드 그대로 유지)
    target_face_height = h * 0.28
    base_scale = target_face_height / current_face_height
    scale = base_scale * (0.70 if is_profile else 1.0)
    
    # [4] 변환 행렬 생성 (기존 코드 그대로 유지)
    M = cv2.getRotationMatrix2D(tuple(nose_bridge), angle, scale)
    
    # [5] 라인 고정 로직 (기존 코드 그대로 유지)
    target_brow_y = h * 0.35
    target_chin_y = h * 0.61 if is_profile else h * 0.65
    
    curr_bridge_trans = M @ np.array([nose_bridge[0], nose_bridge[1], 1])
    
    M[0, 2] += (w * 0.5 - curr_bridge_trans[0])
    M[1, 2] += (h * 0.40 - curr_bridge_trans[1])
    
    # [6] 측면 전용 추가 수직 보정 (기존 코드 그대로 유지)
    if is_profile:
        M[1, 2] -= (h * 0.05)

    # --- 수정 사항: 빈 공간 최소화 및 주변 색 확장 ---
    # 1. 사진이 회전/축소된 후의 실제 범위를 계산하여 캔버스 크기 결정
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    rect_trans = cv2.transform(np.array([rect]), M)[0]
    
    min_x, min_y = np.min(rect_trans, axis=0)
    max_x, max_y = np.max(rect_trans, axis=0)
    
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))
    
    # 2. 이미지가 캔버스 밖으로 잘리지 않게 이동값(min_x, min_y) 보정
    M[0, 2] -= min_x
    M[1, 2] -= min_y

    # 3. 빈 공간을 주변 색으로 늘리는 BORDER_REPLICATE 적용
    aligned_img = cv2.warpAffine(
        img_array, 
        M, 
        (new_w, new_h), 
        borderMode=cv2.BORDER_REPLICATE, 
        flags=cv2.INTER_LINEAR
    )
    
    return aligned_img

# --- 아래 스트림릿 인터페이스는 기존과 동일 ---
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
