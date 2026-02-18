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

st.set_page_config(page_title="Universal Face Aligner", layout="wide")
st.title("📸 AI 전각도 얼굴 정렬기 (측면 대응)")

if 'engine' not in st.session_state:
    st.session_state.engine = load_ai_engine()

face_mesh = st.session_state.engine

def process_universal_align(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # [수직 기준점 추출]
    # 10번: 이마 끝(Hairline), 152번: 턱 끝(Chin)
    # 6번: 미간(Bridge of nose), 1번: 코끝(Tip of nose)
    forehead = landmarks[10]
    chin = landmarks[152]
    nose_bridge = landmarks[6]
    
    # 1. 얼굴의 수직 길이 계산 (이마~턱)
    # 측면으로 돌아가도 수직 길이는 상대적으로 일정하게 유지됩니다.
    face_height_pixel = np.sqrt(((forehead.x - chin.x) * w)**2 + ((forehead.y - chin.y) * h)**2)
    
    # 2. 기준 배율 설정 (사진 짧은 변의 45%를 얼굴 수직 길이로 고정)
    # 이 수치를 조절하여 모든 사진의 얼굴 크기를 통일합니다.
    target_face_height = min(h, w) * 0.45
    scale = target_face_height / face_height_pixel
    
    # 3. 리사이즈
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 4. 중심점 설정 (미간 위치를 기준으로 정렬)
    # 측면 사진에서도 미간(nose_bridge)은 얼굴의 중심 축 역할을 합니다.
    center_y = int(nose_bridge.y * new_h)
    center_x = int(nose_bridge.x * new_w)
    
    # 5. 원본 크기 캔버스에 안착 (비율 유지)
    final_img = np.zeros((h, w, 3), dtype=np.uint8)
    half_h, half_w = h // 2, w // 2
    
    y1, y2 = center_y - half_h, center_y + half_h
    x1, x2 = center_x - half_w, center_x + half_w
    
    src_y1, src_y2 = max(0, y1), min(new_h, y2)
    src_x1, src_x2 = max(0, x1), min(new_w, x2)
    
    dst_y1, dst_x1 = max(0, -y1), max(0, -x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    
    final_img[dst_y1:dst_y2, dst_x1:dst_x2] = img_resized[src_y1:src_y2, src_x1:src_x2]
    
    return final_img

uploaded_files = st.file_uploader("사진들을 업로드하세요", accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(3)
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        result = process_universal_align(img_array)
        
        with cols[idx % 3]:
            if result is not None:
                st.image(result, caption=f"수직정렬 완료: {uploaded_file.name}", use_column_width=True)
                res_img = Image.fromarray(result)
                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button("💾 다운로드", buf.getvalue(), f"aligned_{uploaded_file.name}", "image/png", key=f"dl_{idx}")
