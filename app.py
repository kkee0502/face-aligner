import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# [1. AI 엔진 로드] - 배포 환경에 최적화된 경로로 설정
def load_ai_engine():
    try:
        import mediapipe as mp
        from mediapipe.solutions import face_mesh as mp_face_mesh
        return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    except:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

st.set_page_config(page_title="Universal Face Aligner", layout="wide")
st.title("📸 AI 전각도 얼굴 정렬기")
st.write("정면, 미소, 측면 사진까지 얼굴 크기와 높이를 일정하게 맞춥니다.")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()

face_mesh = st.session_state.engine

# [2. 핵심 정렬 함수]
def align_face_universal(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    
    # AI 인식 (BGR 변환 필요)
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # 기준점 추출: 측면에서도 변하지 않는 수직축 기준
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])   # 왼쪽 눈
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h]) # 오른쪽 눈
    nose_bridge = np.array([landmarks[6].x * w, landmarks[6].y * h]) # 미간 (중심축)
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])    # 턱끝
    
    # 1. 회전 각도 계산 (두 눈의 수평 유지)
    dY = r_eye[1] - l_eye[1]
    dX = r_eye[0] - l_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # 2. 배율 계산 (측면 대응 핵심)
    # 가로(눈 사이 거리) 대신 수직(미간~턱) 길이를 기준으로 배율을 설정합니다.
    face_height_pixel = np.sqrt((nose_bridge[0] - chin[0])**2 + (nose_bridge[1] - chin[1])**2)
    
    # 사진 높이의 35%를 얼굴 수직 길이로 고정 (모든 사진의 얼굴 크기 통일)
    target_face_height = h * 0.35 
    scale = target_face_height / face_height_pixel
    
    # 3. 유사 변환 행렬 생성 (왜곡 없이 회전+배율+이동)
    # 중심점은 얼굴의 기둥인 미간으로 잡습니다.
    M = cv2.getRotationMatrix2D(tuple(nose_bridge), angle, scale)
    
    # 4. 위치 고정 (사진의 수평 중앙, 수직 40% 지점에 미간 고정)
    tX = w * 0.5
    tY = h * 0.40
    M[0, 2] += (tX - nose_bridge[0])
    M[1, 2] += (tY - nose_bridge[1])
    
    # 최종 변환 실행 (검은 여백 처리)
    aligned_img = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return aligned_img

# [3. UI 및 파일 처리]
uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(3)
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        
        # 정렬 프로세스 실행
        result = align_face_universal(img_array)
        
        with cols[idx % 3]:
            if result is not None:
                # 결과 출력 (비율 유지)
                st.image(result, caption=f"정렬됨: {uploaded_file.name}", use_column_width=True)
                
                # 다운로드 버튼
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button(
                    label="💾 다운로드",
                    data=buf.getvalue(),
                    file_name=f"aligned_{uploaded_file.name}",
                    mime="image/png",
                    key=f"dl_{idx}"
                )
            else:
                st.warning(f"{uploaded_file.name}: 얼굴 인식 실패")
