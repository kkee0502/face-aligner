import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def load_ai_engine():
    import mediapipe as mp
    from mediapipe.solutions import face_mesh as mp_face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

st.set_page_config(page_title="Portrait Fit Aligner", layout="wide")
st.title("🔍 얼굴 비율 최적화(75%) & 정면 기준 통합 정렬기")
st.write("첫 번째 사진(정면)을 기준으로 얼굴이 화면의 약 3/4을 차지하도록 조절하여 정렬합니다.")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

if 'base_face_metrics' not in st.session_state:
    st.session_state.base_face_metrics = None

def align_and_fit(img_array, is_first_image):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # 주요 포인트: 눈썹중앙(8), 미간(6), 입술중앙선(0), 동공(468, 473)
    brow = np.array([landmarks[8].x * w, landmarks[8].y * h])
    bridge = np.array([landmarks[6].x * w, landmarks[6].y * h])
    lip = np.array([landmarks[0].x * w, landmarks[0].y * h])
    l_pupil = np.array([landmarks[468].x * w, landmarks[468].y * h])
    r_pupil = np.array([landmarks[473].x * w, landmarks[473].y * h])
    
    # 수평 각도
    angle = np.degrees(np.arctan2(r_pupil[1] - l_pupil[1], r_pupil[0] - l_pupil[0]))

    # [로직] 정면 기준 스케일링 + 비율 최적화 (핵심 수정)
    current_v_dist = np.linalg.norm(brow - lip) # 현재 사진의 눈썹-입술 거리

    # 목표 비율 설정: '눈썹~입술' 거리가 전체 화면 높이의 약 40%가 되도록 설정
    # 이렇게 하면 머리 위와 턱 아래 여백이 자연스럽게 확보되어 얼굴이 화면의 약 75% 정도 차지하게 됩니다.
    target_ratio = 0.40 

    if is_first_image:
        # 첫 번째 사진(정면)의 실제 거리를 기준값으로 저장
        st.session_state.base_face_metrics = {
            'v_dist': current_v_dist,
            'bridge_y_ratio': 0.45 # 미간을 화면의 45% 높이에 배치 (안정적인 구도)
        }
        # 첫 사진의 스케일 계산
        scale = (h * target_ratio) / current_v_dist
    else:
        # 측면 사진의 경우, 정면의 절대 픽셀 거리에 맞춘 뒤 목표 비율 적용
        if st.session_state.base_face_metrics:
            base_v_dist = st.session_state.base_face_metrics['v_dist']
            # 현재 얼굴을 정면 크기로 맞추는 스케일 * 정면을 목표 비율로 맞추는 스케일
            scale = (base_v_dist / current_v_dist) * ((h * target_ratio) / base_v_dist)
        else:
            scale = (h * target_ratio) / current_v_dist # 기준 없으면 자체 비율 적용

    # 변환 행렬 생성 (미간 중심)
    M = cv2.getRotationMatrix2D(tuple(bridge), angle, scale)

    # 미간 위치 강제 고정 (정면 기준 비율 적용)
    t_bridge = M @ np.array([bridge[0], bridge[1], 1])
    target_y = st.session_state.base_face_metrics['bridge_y_ratio'] * h if st.session_state.base_face_metrics else h * 0.45
    
    M[0, 2] += (w * 0.5 - t_bridge[0])  # 가로 중앙
    M[1, 2] += (target_y - t_bridge[1]) # 세로 정면 기준 고정

    # 이미지 워핑 (여백은 가장자리 픽셀 복사로 자연스럽게 채움)
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
                # result에서 직접 높이(res_h)와 너비(res_w)를 가져와 에러 방지
                res_h, res_w = result.shape[:2]
                
                if show_guide:
                    # 최적화된 비율에 맞춘 가이드 라인 (눈썹, 미간, 입술)
                    # 눈썹(brow_y) = bridge_y - (target_ratio * 0.5 * h) 근처
                    # 입술(lip_y) = bridge_y + (target_ratio * 0.5 * h) 근처
                    # 수학적 비례에 따라 계산된 가이드라인 위치
                    bridge_y_ratio = st.session_state.base_face_metrics['bridge_y_ratio'] if st.session_state.base_face_metrics else 0.45
                    
                    # 눈썹(초록), 미간(빨강), 입술(하늘)
                    # 비율은 미간 고정점(0.45)을 기준으로 눈썹~입술 거리(0.40)의 절반씩 가감
                    guide_y_ratios = [bridge_y_ratio - target_ratio/2, bridge_y_ratio, bridge_y_ratio + target_ratio/2]
                    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255)]
                    
                    for ratio, color in zip(guide_y_ratios, colors):
                        y_pos = int(res_h * ratio)
                        cv2.line(result, (0, y_pos), (res_w, y_pos), color, 2)
                
                st.image(result, caption=f"Aligned: {uploaded_file.name}", use_column_width=True)
                
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾", buf.getvalue(), f"fit_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
