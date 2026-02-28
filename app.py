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

st.set_page_config(page_title="Multi-Anchor Aligner", layout="wide")
st.title("📸 4대 핵심 포인트 정밀 정렬기")
st.write("동공, 귀, 코끝, 입술 라인을 기준으로 모든 사진을 표준화합니다.")

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
    
    # [1] 핵심 앵커 포인트 추출
    # 동공 중앙 (468: 왼쪽, 473: 오른쪽)
    l_pupil = np.array([landmarks[468].x * w, landmarks[468].y * h])
    r_pupil = np.array([landmarks[473].x * w, landmarks[473].y * h])
    pupil_center = (l_pupil + r_pupil) / 2
    
    # 귀 (Tragus) (234: 왼쪽, 454: 오른쪽)
    l_ear = np.array([landmarks[234].x * w, landmarks[234].y * h])
    r_ear = np.array([landmarks[454].x * w, landmarks[454].y * h])
    
    # 코끝 (1번) 및 입술 상단 중앙 (0번)
    nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])
    lip_top = np.array([landmarks[0].x * w, landmarks[0].y * h])

    # [2] 정밀 수평 및 스케일 계산
    # 동공 간의 기울기를 기준으로 각도 계산
    angle = np.degrees(np.arctan2(r_pupil[1] - l_pupil[1], r_pupil[0] - l_pupil[0]))
    
    # '동공 중앙 ~ 입술 상단' 거리를 기준으로 전체 얼굴 크기 표준화 (화면 높이의 25%)
    current_dist = np.linalg.norm(pupil_center - lip_top)
    target_dist = h * 0.25
    scale = target_dist / current_dist

    # [3] 변환 행렬 생성 (회전 중심: 코끝)
    M = cv2.getRotationMatrix2D(tuple(nose_tip), angle, scale)

    # [4] 위치 강제 고정 (Line-Lock)
    # 코끝을 화면 중앙(50%), 세로 55% 지점에 고정
    t_nose = M @ np.array([nose_tip[0], nose_tip[1], 1])
    M[0, 2] += (w * 0.5 - t_nose[0])
    M[1, 2] += (h * 0.55 - t_nose[1])

    # [5] 이미지 생성 및 여백 확장
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return aligned_img

# --- UI 레이아웃 ---
uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    show_guide = st.checkbox("동공-귀-코끝-입술 기준선 표시", value=True)
    cols = st.columns(len(uploaded_files))
    
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_precise_line_lock(img_array)
        
        with cols[idx]:
            if result is not None:
                if show_guide:
                    res_h, res_w = result.shape[:2]
                    # 동공(0.35), 귀(0.42), 코끝(0.55), 입술(0.65) 타겟 비율
                    guide_lines = [0.35, 0.42, 0.55, 0.65] 
                    colors = [(255, 255, 0), (255, 0, 255), (0, 255, 0), (0, 255, 255)] 
                    for line_y, color in zip(guide_lines, colors):
                        y_coord = int(res_h * line_y)
                        cv2.line(result, (0, y_coord), (res_w, y_coord), color, 2)
                
                st.image(result, caption=f"정밀 정렬: {uploaded_file.name}", use_column_width=True)
                
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾", buf.getvalue(), f"pro_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
