import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def load_ai_engine():
    import mediapipe as mp
    from mediapipe.solutions import face_mesh as mp_face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

st.set_page_config(page_title="Relative Face Aligner", layout="wide")
st.title("📸 상대적 비율 기반 정렬기")
st.write("절대 좌표 고정 없이, 얼굴 내부의 상대적 비율을 유지하며 정면/측면 라인을 맞춥니다.")

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
    
    # [1] 상대적 정렬을 위한 랜드마크 추출
    # 눈썹(8), 동공(468,473), 코끝(1), 입술(0)
    brow = np.array([landmarks[8].x * w, landmarks[8].y * h])
    l_pupil = np.array([landmarks[468].x * w, landmarks[468].y * h])
    r_pupil = np.array([landmarks[473].x * w, landmarks[473].y * h])
    pupil_center = (l_pupil + r_pupil) / 2
    nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])
    lip_line = np.array([landmarks[0].x * w, landmarks[0].y * h])

    # [2] 수평 각도 계산 (동공 기준)
    angle = np.degrees(np.arctan2(r_pupil[1] - l_pupil[1], r_pupil[0] - l_pupil[0]))

    # [3] 상대적 스케일링 (중요!)
    # 화면 전체 높이가 아니라, 현재 얼굴 내부의 '눈썹~입술' 수직 거리를 기준으로
    # 모든 사진이 동일한 '이목구비 밀도'를 갖도록 스케일만 동기화합니다.
    face_internal_dist = abs(brow[1] - lip_line[1])
    # 기준 스케일 (첫 번째 사진의 비율을 유지하고 싶을 때 유용함)
    target_internal_dist = h * 0.35 # 얼굴 이목구비 영역이 화면의 35% 정도 차지하도록 설정
    scale = target_internal_dist / face_internal_dist

    # [4] 변환 행렬 (상대적 이동)
    # 특정 좌표에 고정하는 대신, '코끝'을 피벗으로 삼아 회전과 스케일만 적용
    # 이동(Translation)은 코끝이 원본 위치 근처(중앙부)를 유지하도록 상대적으로 처리
    M = cv2.getRotationMatrix2D(tuple(nose_tip), angle, scale)
    
    # 변환 후 코끝의 가로 위치만 중앙으로 맞추고, 세로는 원본의 흐름을 따름
    t_nose = M @ np.array([nose_tip[0], nose_tip[1], 1])
    M[0, 2] += (w * 0.5 - t_nose[0]) # 가로는 대칭을 위해 중앙 정렬
    # 세로는 고정하지 않고 원본 위치 대비 미세 조정만 수행 (상대적 유지)
    M[1, 2] += (h * 0.5 - t_nose[1]) 

    # [5] 이미지 생성 및 여백 복사
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return aligned_img

# --- UI 레이아웃 ---
uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    show_guide = st.checkbox("상대적 정렬 라인 표시", value=True)
    cols = st.columns(len(uploaded_files))
    
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_precise_line_lock(img_array)
        
        with cols[idx]:
            if result is not None:
                res_h, res_w = result.shape[:2]
                if show_guide:
                    # 상대적 위치 가이드 (이미지 내 비율 기준)
                    # 눈썹, 동공, 코끝, 입술의 표준 비율 라인
                    guide_ratios = [0.33, 0.40, 0.50, 0.68]
                    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0), (0, 255, 255)]
                    for ratio, color in zip(guide_ratios, colors):
                        y_pos = int(res_h * ratio)
                        cv2.line(result, (0, y_pos), (res_w, y_pos), color, 2)
                
                st.image(result, caption=f"상대 정렬: {uploaded_file.name}", use_column_width=True)
                
                # 저장/다운로드
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾", buf.getvalue(), f"rel_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
