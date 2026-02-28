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
    
    # [1] 정밀 포인트 추출 (눈썹, 미간, 턱) - 수정 없음
    brow_y = (landmarks[105].y + landmarks[334].y) / 2 * h
    nose_bridge = np.array([landmarks[6].x * w, landmarks[6].y * h]) 
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    
    # 눈 수평 각도 - 수정 없음
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

    # [2] 얼굴 높이 계산 (미간 ~ 턱) - 수정 없음
    current_face_height = np.sqrt((nose_bridge[0] - chin[0])**2 + (nose_bridge[1] - chin[1])**2)
    
    # 측면 판별 (눈 가로 길이 비율) - 수정 없음
    eye_dist = np.sqrt((r_eye[0]-l_eye[0])**2 + (r_eye[1]-l_eye[1])**2)
    side_ratio = eye_dist / current_face_height
    is_profile = side_ratio < 0.50  # 값이 작을수록 완전 측면
    
    # [3] 동적 스케일링 설정 (핵심 수정 사항)
    # 측면 여부에 상관없이 얼굴의 수직 길이(미간~턱)가 전체 높이의 30%가 되도록 스케일을 잡습니다.
    # 이렇게 하면 측면 사진이 과하게 커지는 현상이 차단됩니다.
    target_face_height = h * 0.30
    scale = target_face_height / current_face_height
    
    # [4] 변환 행렬 생성 (미간 중심) - 수정 없음
    M = cv2.getRotationMatrix2D(tuple(nose_bridge), angle, scale)
    
    # [5] 위치 고정 로직 (Line-Lock) - 수정 없음
    # 모든 사진의 미간을 가로 50%, 세로 40% 지점으로 '못박기'.
    curr_bridge_trans = M @ np.array([nose_bridge[0], nose_bridge[1], 1])
    
    M[0, 2] += (w * 0.5 - curr_bridge_trans[0])
    M[1, 2] += (h * 0.40 - curr_bridge_trans[1])
    
    # [6] 이미지 생성 (여백 처리 방식 변경)
    # 기존 코드에서는 borderMode=cv2.BORDER_CONSTANT로 검정색 여백이 생겼습니다.
    # 이를 borderMode=cv2.BORDER_REPLICATE로 변경하여, 가장자리 픽셀을 복사해 여백을 채웁니다.
    # 이를 통해 여백이 자연스럽게 배경색과 일치하게 됩니다.
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE, borderValue=(0,0,0))
    
    return aligned_img

uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    # 가이드 라인 표시 여부 - 사용자 가시성을 위해 추가
    show_guide = st.checkbox("가이드 라인 표시 (정렬 확인용)", value=True)
    
    cols = st.columns(len(uploaded_files))
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_precise_line_lock(img_array)
        
        with cols[idx]:
            if result is not None:
                # 가이드라인 그리기 (눈썹 35%, 턱 65% 지점) - 사용자 가시성을 위해 추가
                if show_guide:
                    h_res, w_res, _ = result.shape
                    cv2.line(result, (0, int(h_res*0.35)), (w_res, int(h_res*0.35)), (255, 0, 0), 2)
                    cv2.line(result, (0, int(h_res*0.65)), (w_res, int(h_res*0.65)), (255, 0, 0), 2)
                
                st.image(result, caption=f"라인 동기화: {uploaded_file.name}", use_column_width=True)
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾 다운로드", buf.getvalue(), f"locked_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
