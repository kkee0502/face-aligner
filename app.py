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
st.title("📸 4점 통합 라인 고정 정렬기")
st.write("정수리-눈썹-미간-턱끝의 수직 위치를 모든 사진에서 강제 일치시킵니다.")

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
    
    # [1] 4대 핵심 포인트 추출 (정렬 로직용)
    # 정수리(10번), 미간(6번), 턱끝(152번)
    top_head = np.array([landmarks[10].x * w, landmarks[10].y * h])
    bridge = np.array([landmarks[6].x * w, landmarks[6].y * h])
    
    # 측모 턱끝 인식 보정 (152번 외에 하단 윤곽 199, 200번 조합)
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    is_profile = (np.linalg.norm(l_eye - r_eye) / h) < 0.15 # 눈 거리가 좁으면 측면으로 판단
    
    chin_x = landmarks[152].x * w
    if is_profile:
        # 측면에서는 턱이 들리거나 처지는 것을 방지하기 위해 가중치 보정
        chin_y = (landmarks[152].y * 0.5 + landmarks[199].y * 0.25 + landmarks[200].y * 0.25) * h
    else:
        chin_y = landmarks[152].y * h
    chin = np.array([chin_x, chin_y])

    # [2] 수평 각도 계산
    angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

    # [3] 4점 기준 통합 스케일 계산
    # '정수리 ~ 턱끝'의 전체 길이를 화면 높이의 60%로 강제 고정
    current_full_len = np.linalg.norm(top_head - chin)
    target_full_len = h * 0.60
    scale = target_full_len / current_full_len

    # [4] 변환 행렬 생성 (회전 중심: 미간)
    M = cv2.getRotationMatrix2D(tuple(bridge), angle, scale)

    # [5] 4점 위치 강제 고정 (Translation)
    # 미간(Bridge)을 y=0.45 지점에 고정하면 비율에 따라 정수리/눈썹/턱이 자동 정렬됨
    t_bridge = M @ np.array([bridge[0], bridge[1], 1])
    M[0, 2] += (w * 0.5 - t_bridge[0])  # 가로 중앙
    M[1, 2] += (h * 0.45 - t_bridge[1]) # 세로 미간 고정

    # [6] 이미지 생성 및 여백 복사 (Border Replicate)
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return aligned_img

# --- UI 부분 ---
uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    show_guide = st.checkbox("4대 기준선 표시 (정수리-눈썹-미간-턱)", value=True)
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
                    # 정수리(0.23), 눈썹(0.38), 미간(0.45), 턱끝(0.83) - 타겟 비율에 맞춤
                    guide_lines = [0.23, 0.38, 0.45, 0.83] 
                    colors = [(255, 255, 0), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # 노랑, 초록, 빨강, 하늘
                    for line_y, color in zip(guide_lines, colors):
                        y_coord = int(res_h * line_y)
                        cv2.line(result, (0, y_coord), (res_w, y_coord), color, 2)
                
                st.image(result, caption=f"정렬 완료: {uploaded_file.name}", use_column_width=True)
                
                # 저장 로직
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾", buf.getvalue(), f"locked_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
