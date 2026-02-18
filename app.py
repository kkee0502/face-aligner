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

st.set_page_config(page_title="Pixel-Perfect Aligner", layout="wide")
st.title("📸 초정밀 얼굴 위치 고정기")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

def align_face_perfect(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # [1] 현재 사진의 기준점 (눈 중심, 코끝)
    # 정면/측면 모두에서 가장 안정적인 점 3개를 고릅니다.
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
    
    src_pts = np.float32([l_eye, r_eye, nose])

    # [2] 우리가 원하는 '정답 위치' (Target)
    # 결과물 이미지 안에서 눈과 코가 위치해야 할 좌표를 아예 지정합니다.
    # 예: 가로 1/3, 2/3 지점에 눈을 두고, 중앙에 코를 둡니다.
    dst_pts = np.float32([
        [w * 0.35, h * 0.45], # 왼쪽 눈 고정석
        [w * 0.65, h * 0.45], # 오른쪽 눈 고정석
        [w * 0.50, h * 0.60]  # 코끝 고정석
    ])

    # [3] 아핀 변환 행렬 계산 (삼각형 매칭)
    # src_pts를 dst_pts로 만들기 위한 회전/배율/이동 값을 한 번에 계산합니다.
    matrix = cv2.getAffineTransform(src_pts, dst_pts)
    
    # [4] 이미지 변형 실행
    # 이제 모든 사진은 강제로 dst_pts 위치에 눈과 코가 놓이게 됩니다.
    aligned_img = cv2.warpAffine(img_array, matrix, (w, h))
    
    return aligned_img

uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(3)
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_face_perfect(img_array)
        
        with cols[idx % 3]:
            if result is not None:
                st.image(result, caption=f"완전고정: {uploaded_file.name}", use_column_width=True)
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾 다운로드", buf.getvalue(), f"fixed_{uploaded_file.name}", "image/png", key=f"fixed_{idx}")
