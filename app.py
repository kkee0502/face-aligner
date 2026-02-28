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

st.set_page_config(page_title="Line-Lock Aligner Pro", layout="wide")
st.title("📸 4점 입술-라인 고정 정렬기")
st.write("정수리-눈썹-미간-입술 아래 경계를 모든 사진에서 수학적으로 강제 일치시킵니다.")

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
    
    # [1] 4대 핵심 포인트 추출 (입술 아래 경계 기준)
    # 정수리(10번), 미간(6번), 입술 아래 중앙 경계(17번)
    top_head = np.array([landmarks[10].x * w, landmarks[10].y * h])
    bridge = np.array([landmarks[6].x * w, landmarks[6].y * h])
    lip_bottom = np.array([landmarks[17].x * w, landmarks[17].y * h])
    
    # 눈 수평 각도 (33번, 263번)
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

    # [2] 통합 스케일 계산 (정수리 ~ 입술 아래 경계 기준)
    # 모든 사진에서 '정수리 ~ 입술 아래' 거리를 화면 높이의 50%로 강제 고정
    current_dist = np.linalg.norm(top_head - lip_bottom)
    target_dist = h * 0.50
    scale = target_dist / current_dist

    # [3] 변환 행렬 생성 (회전 중심: 미간)
    M = cv2.getRotationMatrix2D(tuple(bridge), angle, scale)

    # [4] 위치 강제 고정 (Translation)
    # 미간(Bridge)을 화면의 y=0.45(45% 지점)에 '못박기'
    t_bridge = M @ np.array([bridge[0], bridge[1], 1])
    M[0, 2] += (w * 0.5 - t_bridge[0])  # 가로 중앙 정렬
    M[1, 2] += (h * 0.45 - t_bridge[1]) # 세로 미간 고정

    # [5] 이미지 생성 및 여백 복사 (Border Replicate)
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return aligned_img

# --- UI 레이아웃 ---
uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    show_guide = st.checkbox("4대 기준선 표시 (정수리-눈썹-미간-입술아래)", value=True)
    cols = st.columns(len(uploaded_files))
    
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_precise_line_lock(img_array)
        
        with cols[idx]:
            if result is not None:
                if show_guide:
                    # 에러 수정: result의 shape를 직접 참조하여 선을 긋습니다.
                    res_h, res_w = result.shape[:2]
                    # 정수리(0.24), 눈썹(0.38), 미간(0.45), 입술아래(0.74) - 타겟 비율
                    guide_lines = [0.24, 0.38, 0.45, 0.74] 
                    colors = [(255, 255, 0), (0, 255, 0), (255, 0, 0), (0, 255, 255)] 
                    for line_y, color in zip(guide_lines, colors):
                        y_coord = int(res_h * line_y)
                        cv2.line(result, (0, y_coord), (res_w, y_coord), color, 2)
                
                st.image(result, caption=f"입술기준 정렬: {uploaded_file.name}", use_column_width=True)
                
                # 저장/다운로드 로직
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾", buf.getvalue(), f"lip_fixed_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
