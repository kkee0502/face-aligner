import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def load_ai_engine():
    import mediapipe as mp
    from mediapipe.solutions import face_mesh as mp_face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

st.set_page_config(page_title="Cross-View Aligner", layout="wide")
st.title("📸 정면-측면 통합 라인 정렬기")
st.write("정면과 측면의 이목구비 높이를 수학적으로 강제 일치시킵니다.")

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
    
    # [1] 정면/측면 공통 불변 포인트 추출
    # 동공(468, 473), 코끝(1), 입술중앙(0), 귀(234 or 454)
    l_pupil = np.array([landmarks[468].x * w, landmarks[468].y * h])
    r_pupil = np.array([landmarks[473].x * w, landmarks[473].y * h])
    pupil_y_avg = (l_pupil[1] + r_pupil[1]) / 2
    
    nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])
    lip_top = np.array([landmarks[0].x * w, landmarks[0].y * h])
    
    # 귀(Tragus) 포인트: 측면 판별에 따라 적절한 쪽 선택
    ear_l = np.array([landmarks[234].x * w, landmarks[234].y * h])
    ear_r = np.array([landmarks[454].x * w, landmarks[454].y * h])
    # 더 카메라에 가까운(화면 끝에 가까운) 귀를 선택
    ear_y = ear_l[1] if abs(ear_l[0] - w/2) > abs(ear_r[0] - w/2) else ear_r[1]

    # [2] 정면-측면 통합 스케일 계산 (핵심 수정)
    # 가로 거리는 회전 시 변하므로 절대 사용 금지.
    # '동공 높이 ~ 입술 높이'의 수직 차이만 사용하여 스케일 결정
    current_v_dist = abs(pupil_y_avg - lip_top[1])
    target_v_dist = h * 0.22 # 전체 화면의 22%로 얼굴 높이 고정
    scale = target_v_dist / current_v_dist

    # [3] 수평 각도 계산 (동공 기준)
    angle = np.degrees(np.arctan2(r_pupil[1] - l_pupil[1], r_pupil[0] - l_pupil[0]))

    # [4] 변환 행렬 생성 (회전 중심: 코끝)
    M = cv2.getRotationMatrix2D(tuple(nose_tip), angle, scale)

    # [5] 4점 라인 고정 (Line-Lock)
    # 코끝(Nose Tip)을 모든 사진에서 y=0.55 (55% 지점)에 강제 고정
    t_nose = M @ np.array([nose_tip[0], nose_tip[1], 1])
    M[0, 2] += (w * 0.5 - t_nose[0])
    M[1, 2] += (h * 0.55 - t_nose[1])

    # [6] 이미지 워핑 및 여백 처리
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return aligned_img

# --- UI 부분 ---
uploaded_files = st.file_uploader("정면과 측면 사진을 함께 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    show_guide = st.checkbox("동공-귀-코끝-입술 통합 라인 표시", value=True)
    cols = st.columns(len(uploaded_files))
    
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = align_precise_line_lock(img_array)
        
        with cols[idx]:
            if result is not None:
                res_h, res_w = result.shape[:2]
                if show_guide:
                    # 정면/측면 공통 타겟 높이 (비율 고정)
                    # 동공(0.33), 귀(0.40), 코끝(0.55), 입술(0.66)
                    guide_y = [0.33, 0.40, 0.55, 0.66]
                    colors = [(255,255,0), (255,0,255), (0,255,0), (0,255,255)]
                    for y_ratio, color in zip(guide_y, colors):
                        y_pos = int(res_h * y_ratio)
                        cv2.line(result, (0, y_pos), (res_w, y_pos), color, 2)
                
                st.image(result, caption=uploaded_file.name, use_column_width=True)
                
                # 저장용
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾", buf.getvalue(), f"fixed_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
