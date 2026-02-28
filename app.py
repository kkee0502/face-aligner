def align_precise_line_lock(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # [1] 정밀 포인트 추출 (기존 유지)
    brow_y = (landmarks[105].y + landmarks[334].y) / 2 * h
    nose_bridge = np.array([landmarks[6].x * w, landmarks[6].y * h]) 
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

    # [2] 얼굴 높이 및 측면 판별 (기존 유지)
    current_face_height = np.sqrt((nose_bridge[0] - chin[0])**2 + (nose_bridge[1] - chin[1])**2)
    eye_dist = np.sqrt((r_eye[0]-l_eye[0])**2 + (r_eye[1]-l_eye[1])**2)
    side_ratio = eye_dist / current_face_height
    is_profile = side_ratio < 0.50 
    
    # [3] 배율 설정 (기존 유지)
    target_face_height = h * 0.28
    base_scale = target_face_height / current_face_height
    scale = base_scale * (0.70 if is_profile else 1.0)
    
    # [4] 변환 행렬 생성 (미간 중심)
    M = cv2.getRotationMatrix2D(tuple(nose_bridge), angle, scale)

    # --- 추가된 크롭 최적화 로직 ---
    # 원본 이미지의 네 모서리가 변환 후 어디에 위치하는지 계산
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    rect_trans = cv2.transform(np.array([rect]), M)[0]
    
    # 변환된 이미지의 실제 가로/세로 범위를 구함
    min_x, min_y = np.min(rect_trans, axis=0)
    max_x, max_y = np.max(rect_trans, axis=0)
    
    # 새로운 캔버스 크기 결정 (소수점 올림 처리)
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))
    # ----------------------------

    # [5] 라인 고정 로직 (기존 비율 유지하되 새로운 크기 기준 적용)
    # 현재 미간 위치가 변환 후 어디로 가는지 확인
    curr_bridge_trans = M @ np.array([nose_bridge[0], nose_bridge[1], 1])
    
    # 새로운 캔버스의 중심 및 수직 40% 지점으로 이동
    # min_x, min_y를 빼주는 이유는 변환된 좌표계의 시작점을 0,0으로 맞추기 위함입니다.
    M[0, 2] += (new_w * 0.5 - curr_bridge_trans[0])
    M[1, 2] += (new_h * 0.40 - curr_bridge_trans[1])
    
    # [6] 측면 전용 추가 수직 보정 (기존 유지)
    if is_profile:
        M[1, 2] -= (new_h * 0.05) 

    # [최종 결과] 주변색으로 채우기(BORDER_REPLICATE) 적용 및 크기 최적화
    aligned_img = cv2.warpAffine(
        img_array, 
        M, 
        (new_w, new_h), 
        borderMode=cv2.BORDER_REPLICATE, 
        flags=cv2.INTER_LINEAR
    )
    
    return aligned_img
