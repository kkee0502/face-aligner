import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def load_ai_engine():
    import mediapipe as mp
    from mediapipe.solutions import face_mesh as mp_face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

st.set_page_config(page_title="Balanced Portrait Aligner", layout="wide")
st.title("📸 여백 최적화 & 정면 기준 통합 정렬기")
st.write("얼굴 주변에 적절한 여백을 두어 답답함 없이 정렬합니다. (첫 사진 기준)")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

# 전역 상수: 이목구비(눈썹~입술)가 화면 높이에서 차지할 비율을 0.32(32%)로 하향 조정
# 이렇게 하면 얼굴 전체가 화면의 약 55~60% 정도를 차지하며 여백이 충분해집니다.
TARGET_FACE_RATIO = 0.32 

if 'base_face_metrics' not in st.session_state:
    st.session_state.base_face_metrics = None

def align_and_fit(img_array, is_first_image):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # 주요 포인트 추출
    brow = np.array([landmarks[8].x * w, landmarks[8].y * h])
    bridge = np.array([landmarks[6].x * w, landmarks[6].y * h])
    lip = np.array([landmarks[0].x * w, landmarks[0].y * h])
    l_pupil = np.array([landmarks[468].x * w, landmarks[468].y * h])
    r_pupil = np.array([landmarks[473].x * w, landmarks[473].y * h])
    
    angle = np.degrees(np.arctan2(r_pupil[1] - l_pupil[1], r_pupil[0] - l_pupil[0]))
    current_v_dist = np.linalg.norm(brow - lip)

    if is_first_image:
        # 첫 사진 기준값 저장 (미간 위치를 0.48로 살짝 내려서 헤드룸 확보)
        st.session_state.base_face_metrics = {
            'v_dist': current_v_dist,
            'bridge_y_ratio': 0.48 
        }
        scale = (h * TARGET_FACE_RATIO) / current_v_dist
    else:
        if st.session_state.base_face_metrics:
            base_v_dist = st.session_state.base_face_metrics['v_dist']
            scale = (base_v_dist / current_v_dist) * ((h * TARGET_FACE_RATIO) / base_v_dist)
        else:
            scale = (h * TARGET_FACE_RATIO) / current_v_dist

    M = cv2.getRotationMatrix2D(tuple(bridge), angle, scale)
    t_bridge = M @ np.array([bridge[0], bridge[1], 1])
    target_y = st.session_state.base_face_metrics['bridge_y_ratio'] * h if st.session_state.base_face_metrics else h * 0.48
    
    M[0, 2] += (w * 0.5 - t_bridge[0])
    M[1, 2] += (target_y - t_bridge[1])

    # 여백 처리는 여전히 자연스럽게 유지
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return aligned_img

# --- UI 부분 ---
uploaded_files = st.file_uploader("정면 사진부터 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    show_guide = st.checkbox("가이드라인 표시 (확인용)", value=True)
    cols = st.columns(len(uploaded_files))
    
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        
        result = align_and_fit(img_array, is_first_image=(idx == 0))
        
        with cols[idx]:
            if result is not None:
                res_h, res_w = result.shape[:2]
                
                if show_guide:
                    b_y_ratio = st.session_state.base_face_metrics['bridge_y_ratio'] if st.session_state.base_face_metrics else 0.48
                    guide_lines = [b_y_ratio - TARGET_FACE_RATIO/2, b_y_ratio, b_y_ratio + TARGET_FACE_RATIO/2]
                    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255)]
                    
                    for r, color in zip(guide_lines, colors):
                        y_pos = int(res_h * r)
                        cv2.line(result, (0, y_pos), (res_w, y_pos), color, 2)
                
                st.image(result, caption=uploaded_file.name, use_column_width=True)
                
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾", buf.getvalue(), f"balanced_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
