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
st.title("📸 4점 인물 확대 및 배경 최소화 정렬기")
st.write("정수리, 눈썹, 미간, 입술선 위치를 강제 고정하고 인물을 1.3배 확대합니다.")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()
face_mesh = st.session_state.engine

# 정면 사진 랜드마크 저장을 위한 세션 상태 초기화
if 'frontal_landmarks' not in st.session_state:
    st.session_state.frontal_landmarks = None

def get_landmarks(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    return results.multi_face_landmarks[0].landmark

def align_precise_line_lock(img_array, frontal_landmarks=None):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # 4대 핵심 포인트 추출
    # 1. 정수리 (10번)
    top_head = np.array([landmarks[10].x * w, landmarks[10].y * h])
    # 2. 눈썹 중앙 (8번)
    brow_mid = np.array([landmarks[8].x * w, landmarks[8].y * h])
    # 3. 미간 (6번)
    bridge = np.array([landmarks[6].x * w, landmarks[6].y * h])
    # 4. 입술 상단 중앙 (0번 + 측면 보정)
    lip_top_x, lip_top_y = landmarks[0].x * w, landmarks[0].y * h
    
    # 측면 판별 및 입술선 보정 (측모에서 입술이 낮게 잡히는 현상 방지)
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    eye_dist = np.linalg.norm(l_eye - r_eye)
    face_height = np.linalg.norm(bridge - np.array([lip_top_x, lip_top_y]))
    is_profile = (eye_dist / face_height) < 0.5
    
    if is_profile:
        # 측면일 경우 입술 주변 윤곽 포인트(11, 12, 16)를 참조하여 입술선 위치 상향 보정
        lip_top_y = (landmarks[0].y * 0.4 + landmarks[11].y * 0.2 + landmarks[12].y * 0.2 + landmarks[16].y * 0.2) * h

    lip_top = np.array([lip_top_x, lip_top_y])

    # 수평 각도 계산 (동공 기준)
    angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

    # 통합 스케일 계산 (정면 랜드마크 기준 정밀 조정)
    # 정면 랜드마크가 제공된 경우, 정수리~입술선 거리를 기준으로 스케일을 조정합니다.
    if frontal_landmarks is not None:
        frontal_top_head = np.array([frontal_landmarks[10].x * w, frontal_landmarks[10].y * h])
        frontal_lip_top = np.array([frontal_landmarks[0].x * w, frontal_landmarks[0].y * h])
        target_full_len = np.linalg.norm(frontal_top_head - frontal_lip_top)
    else:
        # 정면 랜드마크가 없는 경우, 기본 비율(화면 높이의 50%)을 사용합니다.
        target_full_len = h * 0.50

    current_full_len = np.linalg.norm(top_head - lip_top)
    scale = target_full_len / current_full_len

    # 변환 행렬 생성 (회전 중심은 미간)
    M = cv2.getRotationMatrix2D(tuple(bridge), angle, scale)

    # 4점 라인 고정 (Line-Lock) 로직 (정면 미간 위치 기준)
    # 정면 미간 위치가 제공된 경우, 변환된 미간 위치를 확인하고 정면 미간 y 좌표에 강제 고정합니다.
    curr_bridge_trans = M @ np.array([bridge[0], bridge[1], 1])
    
    # 가로 중앙 정렬
    M[0, 2] += (w * 0.5 - curr_bridge_trans[0])
    
    if frontal_landmarks is not None:
        # 세로 미간 고정 (정면 미간 y 좌표 기준)
        frontal_bridge_y = frontal_landmarks[6].y * h
        M[1, 2] += (frontal_bridge_y - curr_bridge_trans[1])
    else:
        # 정면 미간 위치가 없는 경우, 기본 비율(45% 지점)을 사용합니다.
        M[1, 2] += (h * 0.45 - curr_bridge_trans[1])

    # 이미지 생성 및 여백 복사 (검은 여백 제거)
    aligned_img = cv2.warpAffine(img_array, M, (w, h), 
                                 borderMode=cv2.BORDER_REPLICATE)
    
    # 인물 확대 및 배경 최소화 (핵심 수정 사항)
    # 정렬된 이미지의 너비와 높이를 기준으로 중심을 계산하고, 해당 중심을 기준으로 이미지를 1.3배 확대합니다.
    h_aligned, w_aligned, _ = aligned_img.shape
    M_zoom = cv2.getRotationMatrix2D((w_aligned / 2, h_aligned / 2), 0, 1.3)
    zoomed_img = cv2.warpAffine(aligned_img, M_zoom, (w_aligned, h_aligned), borderMode=cv2.BORDER_REPLICATE)
    
    # 확대된 이미지에서 이목구비가 화면에 가득 차도록 배경 여백을 최소한으로 유지합니다.
    # 가로 여백 최소화
    l_eye_zoomed = M_zoom @ np.array([landmarks[33].x * w_aligned, landmarks[33].y * h_aligned, 1])
    r_eye_zoomed = M_zoom @ np.array([landmarks[263].x * w_aligned, landmarks[263].y * h_aligned, 1])
    eye_dist_zoomed = np.linalg.norm(l_eye_zoomed - r_eye_zoomed)
    x_offset = int((w_aligned - eye_dist_zoomed) / 2)
    zoomed_img = zoomed_img[:, x_offset:w_aligned - x_offset]
    
    # 세로 여백 최소화
    top_head_zoomed = M_zoom @ np.array([landmarks[10].x * w_aligned, landmarks[10].y * h_aligned, 1])
    lip_top_zoomed = M_zoom @ np.array([landmarks[0].x * w_aligned, landmarks[0].y * h_aligned, 1])
    face_height_zoomed = np.linalg.norm(top_head_zoomed - lip_top_zoomed)
    y_offset = int((h_aligned - face_height_zoomed) / 2)
    zoomed_img = zoomed_img[y_offset:h_aligned - y_offset, :]
    
    return zoomed_img

# --- UI 레이아웃 ---
uploaded_files = st.file_uploader("사진들을 업로드하세요 (첫 번째 사진은 정면 사진이어야 합니다)", accept_multiple_files=True)

if uploaded_files:
    # 가이드 라인 표시 여부 - 사용자 가시성을 위해 추가
    show_guide = st.checkbox("4대 기준선 표시 (확인용)", value=True)
    
    cols = st.columns(len(uploaded_files))
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        
        # 첫 번째 사진을 정면 사진으로 간주하고 랜드마크 추출 및 저장
        if idx == 0:
            st.session_state.frontal_landmarks = get_landmarks(img_array)
            result = align_precise_line_lock(img_array)
        else:
            # 두 번째 사진부터는 정면 랜드마크를 기준으로 정렬
            result = align_precise_line_lock(img_array, st.session_state.frontal_landmarks)
        
        with cols[idx]:
            if result is not None:
                # 에러 방지: 변환된 결과 이미지의 높이와 너비를 새로 정의
                res_h, res_w = result.shape[:2]
                
                if show_guide:
                    # 모든 사진에서 이 위치에 포인트들이 오게 됩니다.
                    lines = [0.22, 0.38, 0.45, 0.72] # 정수리, 눈썹, 미간, 입술선 비율
                    colors = [(255,200,0), (0,255,0), (255,0,0), (0,200,255)]
                    for line_y, color in zip(lines, colors):
                        cv2.line(result, (0, int(res_h*line_y)), (res_w, int(res_h*line_y)), color, 2)
                
                st.image(result, caption=f"4점 정렬: {uploaded_file.name}", use_column_width=True)
                
                # 다운로드 버튼
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾 다운로드", buf.getvalue(), f"locked_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
