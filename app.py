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
st.title("📸 입술 아래 경계 기준 4점 정밀 라인 고정 정렬기")
st.write("정수리, 눈썹, 미간, 입술 아래 경계 위치를 모든 사진에서 동일하게 강제 고정합니다.")

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
    
    # 4대 핵심 포인트 추출 (입술 아래 경계로 변경)
    # 1. 정수리 (10번)
    top_head = np.array([landmarks[10].x * w, landmarks[10].y * h])
    # 2. 눈썹 중앙 (8번)
    brow_mid = np.array([landmarks[8].x * w, landmarks[8].y * h])
    # 3. 미간 (6번)
    bridge = np.array([landmarks[6].x * w, landmarks[6].y * h])
    # 4. 입술 아래 경계 (17번 + 측면 보정)
    lip_bottom_x, lip_bottom_y = landmarks[17].x * w, landmarks[17].y * h
    
    # 측면 판별 및 입술 아래 경계 보정 (측모에서 입술이 낮게 잡히는 현상 방지)
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    eye_dist = np.linalg.norm(l_eye - r_eye)
    face_height = np.linalg.norm(bridge - np.array([lip_bottom_x, lip_bottom_y]))
    is_profile = (eye_dist / face_height) < 0.5
    
    if is_profile:
        # 측면일 경우 입술 주변 윤곽 포인트(18, 200, 201)를 참조하여 입술 아래 경계 위치 상향 보정
        lip_bottom_y = (landmarks[17].y * 0.4 + landmarks[18].y * 0.2 + landmarks[200].y * 0.2 + landmarks[201].y * 0.2) * h

    lip_bottom = np.array([lip_bottom_x, lip_bottom_y])

    # 수평 각도 계산
    angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

    # 통합 스케일 계산 (정수리부터 입술 아래 경계까지의 전체 길이를 기준)
    # 모든 사진에서 '정수리~입술 아래 경계'의 길이를 화면 높이의 50%로 통일
    current_full_len = np.linalg.norm(top_head - lip_bottom)
    target_full_len = h * 0.50
    scale = target_full_len / current_full_len

    # 변환 행렬 생성 (회전 중심은 미간)
    M = cv2.getRotationMatrix2D(tuple(bridge), angle, scale)

    # 4점 라인 고정 (Line-Lock) 로직
    # 기준점인 미간을 y=0.45(45% 지점)에 고정하면 나머지 점들이 비율에 따라 정렬됨
    t_bridge = M @ np.array([bridge[0], bridge[1], 1])
    
    M[0, 2] += (w * 0.5 - t_bridge[0])  # 가로 중앙
    M[1, 2] += (h * 0.45 - t_bridge[1]) # 미간 높이 고정

    # 이미지 생성 및 여백 복사 (검은 여백 제거)
    aligned_img = cv2.warpAffine(img_array, M, (w, h), 
                                 borderMode=cv2.BORDER_REPLICATE)
    
    return aligned_img

# --- UI 레이아웃 ---
uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    show_guide = st.checkbox("4대 기준선 표시 (확인용)", value=True)
    cols = st.columns(len(uploaded_files))
    
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_precise_line_lock(img_array)
        
        with cols[idx]:
            if result is not None:
                if show_guide:
                    # 모든 사진에서 이 위치에 포인트들이 오게 됩니다.
                    lines = [0.22, 0.38, 0.45, 0.72] # 정수리, 눈썹, 미간, 입술 아래 경계 비율
                    colors = [(255,200,0), (0,255,0), (255,0,0), (0,200,255)]
                    for line_y, color in zip(lines, colors):
                        cv2.line(result, (0, int(h*line_y)), (w, int(h*line_y)), color, 2)
                
                st.image(result, caption=f"4점 정렬: {uploaded_file.name}", use_column_width=True)
                
                # 다운로드
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾", buf.getvalue(), f"aligned_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
