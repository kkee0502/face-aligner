import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def load_ai_engine():
    import mediapipe as mp
    from mediapipe.solutions import face_mesh as mp_face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

st.set_page_config(page_title="Frontal-Base Aligner", layout="wide")
st.title("📸 정면 기준 측모 강제 고정 정렬기")
st.write("첫 번째 사진(정면)의 이목구비 크기를 기준으로 모든 사진을 맞춥니다.")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

# 정면 사진의 기준 데이터 저장을 위한 세션 상태
if 'base_face_metrics' not in st.session_state:
    st.session_state.base_face_metrics = None

def align_to_frontal_base(img_array, is_first_image):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # 앵커 포인트 추출: 눈썹중앙(8), 미간(6), 입술중앙선(0)
    brow = np.array([landmarks[8].x * w, landmarks[8].y * h])
    bridge = np.array([landmarks[6].x * w, landmarks[6].y * h])
    lip = np.array([landmarks[0].x * w, landmarks[0].y * h])
    
    # 수평 각도 (동공 기준)
    l_pupil = np.array([landmarks[468].x * w, landmarks[468].y * h])
    r_pupil = np.array([landmarks[473].x * w, landmarks[473].y * h])
    angle = np.degrees(np.arctan2(r_pupil[1] - l_pupil[1], r_pupil[0] - l_pupil[0]))

    # [핵심 로직] 정면 기준 스케일링
    current_v_dist = np.linalg.norm(brow - lip) # 현재 사진의 눈썹-입술 거리

    if is_first_image:
        # 첫 번째 사진(정면)의 실제 거리를 기준값으로 저장
        st.session_state.base_face_metrics = {
            'v_dist': current_v_dist,
            'bridge_y_ratio': bridge[1] / h  # 정면의 미간 높이 비율 저장
        }
        scale = 1.0
    else:
        # 측면 사진의 경우, 정면의 '눈썹-입술' 길이에 맞춰 자신의 사이즈를 조절
        if st.session_state.base_face_metrics:
            scale = st.session_state.base_face_metrics['v_dist'] / current_v_dist
        else:
            scale = 1.0

    # 변환 행렬 생성 (코끝 대신 미간을 회전축으로 사용)
    M = cv2.getRotationMatrix2D(tuple(bridge), angle, scale)

    # 정면의 미간 높이에 측면의 미간을 강제 고정
    t_bridge = M @ np.array([bridge[0], bridge[1], 1])
    target_y = st.session_state.base_face_metrics['bridge_y_ratio'] * h if st.session_state.base_face_metrics else h * 0.45
    
    M[0, 2] += (w * 0.5 - t_bridge[0])  # 가로 중앙
    M[1, 2] += (target_y - t_bridge[1]) # 세로 정면 기준 고정

    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return aligned_img

uploaded_files = st.file_uploader("사진들을 업로드하세요 (첫 사진이 정면)", accept_multiple_files=True)

if uploaded_files:
    show_guide = st.checkbox("정면 기준 라인 표시", value=True)
    cols = st.columns(len(uploaded_files))
    
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        
        # 첫 번째 사진 여부 전달
        result = align_to_frontal_base(img_array, is_first_image=(idx == 0))
        
        with cols[idx]:
            if result is not None:
                # [에러 수정] result에서 직접 높이(res_h)와 너비(res_w)를 가져옴
                res_h, res_w = result.shape[:2]
                
                if show_guide:
                    # 정면에서 정해진 비율에 맞춰 라인 렌더링
                    guide_y = [0.35, 0.42, 0.45, 0.70] # 눈썹, 동공, 미간, 입술
                    colors = [(0,255,0), (255,255,0), (255,0,0), (0,255,255)]
                    for ratio, color in zip(guide_y, colors):
                        y_pos = int(res_h * ratio)
                        cv2.line(result, (0, y_pos), (res_w, y_pos), color, 2)
                
                st.image(result, caption=f"정렬됨: {uploaded_file.name}", use_column_width=True)
                
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾", buf.getvalue(), f"locked_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
