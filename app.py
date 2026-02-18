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

st.set_page_config(page_title="Machine Frame Aligner", layout="wide")
st.title("📸 기계 프레임 기준 초정밀 정렬기")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

def align_by_machine_frame(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    
    # [1] 기계 장치 인식을 위한 색상 마스크 (녹색 고정핀 기준)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 고정핀의 위치 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 안면 랜드마크도 동시에 추출 (기계가 없을 경우 대비)
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    # 기준점 변수 초기화
    ref_pts = []

    # 기계의 고정핀(녹색)이 발견된 경우
    if len(contours) >= 2:
        # 면적이 큰 순서대로 두 개 선택
        sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        for cnt in sorted_cnts:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                ref_pts.append([cX, cY])
        
        # 좌우 순서로 정렬
        ref_pts = sorted(ref_pts, key=lambda x: x[0])
        
        # 기계 기준 배율 및 각도 계산
        p1, p2 = np.array(ref_pts[0]), np.array(ref_pts[1])
        dX, dY = p2[0] - p1[0], p2[1] - p1[1]
        angle = np.degrees(np.arctan2(dY, dX))
        dist = np.sqrt(dX**2 + dY**2)
        
        # 기계 핀 사이의 거리를 화면 너비의 25%로 고정
        target_dist = w * 0.25
        scale = target_dist / dist
        center = (p1 + p2) / 2
        
        # 변환 행렬
        M_mat = cv2.getRotationMatrix2D(tuple(center), angle, scale)
        
        # 기계 위치를 화면 상단 50% 지점으로 고정
        M_mat[0, 2] += (w * 0.5 - center[0])
        M_mat[1, 2] += (h * 0.5 - center[1])
        
    # 기계가 없거나 인식이 안 된 경우 기존 안면 랜드마크 방식 사용
    elif results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        p1 = np.array([landmarks[33].x * w, landmarks[33].y * h])
        p2 = np.array([landmarks[263].x * w, landmarks[263].y * h])
        
        dX, dY = p2[0] - p1[0], p2[1] - p1[1]
        angle = np.degrees(np.arctan2(dY, dX))
        dist = np.sqrt(dX**2 + dY**2)
        
        target_dist = w * 0.3 # 안면 기준 배율
        scale = target_dist / dist
        center = (p1 + p2) / 2
        
        M_mat = cv2.getRotationMatrix2D(tuple(center), angle, scale)
        M_mat[0, 2] += (w * 0.5 - center[0])
        M_mat[1, 2] += (h * 0.45 - center[1])
    else:
        return img_array

    return cv2.warpAffine(img_array, M_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

uploaded_files = st.file_uploader("기계 촬영 사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(len(uploaded_files))
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_by_machine_frame(img_array)
        
        with cols[idx]:
            if result is not None:
                st.image(result, caption=f"기계기준 정렬: {uploaded_file.name}", use_column_width=True)
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾 다운로드", buf.getvalue(), f"fixed_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
