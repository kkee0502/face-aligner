import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
pip install mediapipie

# [에러 방지] MediaPipe 로드 함수 고도화
def load_ai_engine():
    try:
        import mediapipe as mp
        from mediapipe.solutions import face_mesh as mp_face_mesh
        return mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    except ImportError:
        st.error("Mediapipe 라이브러리가 설치되지 않았습니다. 'pip install mediapipe'를 확인하세요.")
        return None

st.set_page_config(page_title="Professional Portrait Aligner", layout="wide")
st.title("📸 여백 최적화 & 정면 기준 통합 정렬기")
st.write("얼굴 주변에 여유로운 여백을 두어 자연스럽게 정렬합니다. (첫 사진 기준)")

# 전역 상수: 이목구비를 화면 높이의 32%로 설정하여 여백 확보
TARGET_FACE_RATIO = 0.32 

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()

if 'base_face_metrics' not in st.session_state:
    st.session_state.base_face_metrics = None

def align_and_fit(img_array, is_first_image):
    if img_array is None or st.session_state.engine is None:
        return None
    
    h, w, _ = img_array.shape
    results = st.session_state.engine.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # 주요 포인트 추출: 눈썹(8), 미간(6), 입술(0), 동공(468, 473)
    brow = np.array([landmarks[8].x * w, landmarks[8].y * h])
    bridge = np.array([landmarks[6].x * w, landmarks[6].y * h])
    lip = np.array([landmarks[0].x * w, landmarks[0].y * h])
    l_pupil = np.array([landmarks[468].x * w, landmarks[468].y * h])
    r_pupil = np.array([landmarks[473].x * w, landmarks[473].y * h])
    
    # 수평 각도 계산
    angle = np.degrees(np.arctan2(r_pupil[1] - l_pupil[1], r_pupil[0] - l_pupil[0]))
    current_v_dist = np.linalg.norm(brow - lip)

    if is_first_image:
        # 첫 사진 기준값 저장 (미간 높이를 0.48로 설정하여 헤드룸 확보)
        st.session_state.base_face_metrics = {
            'v_dist': current_v_dist,
            'bridge_y_ratio': 0.48 
        }
        scale = (h * TARGET_FACE_RATIO) / current_v_dist
    else:
        # 정면 기준에 맞춰 측면 사진 스케일 조정
        if st.session_state.base_face_metrics:
            base_v_dist = st.session_state.base_face_metrics['v_dist']
            scale = (base_v_dist / current_v_dist) * ((h * TARGET_FACE_RATIO) / base_v_dist)
        else:
            scale = (h * TARGET_FACE_RATIO) / current_v_dist

    # 변환 행렬 생성 (미간 중심)
    M = cv2.getRotationMatrix2D(tuple(bridge), angle, scale)
    t_bridge = M @ np.array([bridge[0], bridge[1], 1])
    
    # 세션 데이터를 안전하게 참조하여 타겟 높이 설정
    target_y_ratio = st.session_state.base_face_metrics['bridge_y_ratio'] if st.session_state.base_face_metrics else 0.48
    target_y = target_y_ratio * h
    
    M[0, 2] += (w * 0.5 - t_bridge[0])  # 가로 중앙
    M[1, 2] += (target_y - t_bridge[1]) # 세로 고정

    # 이미지 워핑 (가장자리 복사로 배경 유지)
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return aligned_img

# --- UI 부분 ---
uploaded_files = st.file_uploader("정면 사진부터 순서대로 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    show_guide = st.checkbox("가이드라인 표시 (눈썹-미간-입술)", value=True)
    cols = st.columns(len(uploaded_files))
    
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        
        # 첫 번째 사진 여부 전달
        result = align_and_fit(img_array, is_first_image=(idx == 0))
        
        with cols[idx]:
            if result is not None:
                res_h, res_w = result.shape[:2]
                
                if show_guide:
                    # 상수를 활용한 안전한 가이드라인 렌더링
                    b_y_ratio = st.session_state.base_face_metrics['bridge_y_ratio'] if st.session_state.base_face_metrics else 0.48
                    guide_lines = [b_y_ratio - TARGET_FACE_RATIO/2, b_y_ratio, b_y_ratio + TARGET_FACE_RATIO/2]
                    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255)]
                    
                    for r, color in zip(guide_lines, colors):
                        y_pos = int(res_h * r)
                        cv2.line(result, (0, y_pos), (res_w, y_pos), color, 2)
                
                st.image(result, caption=f"결과: {uploaded_file.name}", use_column_width=True)
                
                # 저장/다운로드
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾", buf.getvalue(), f"final_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
