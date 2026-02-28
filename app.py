def align_precise_line_lock(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # [1] 정밀 포인트 추출 (원본 유지)
    brow_y = (landmarks[105].y + landmarks[334].y) / 2 * h
    nose_bridge = np.array([landmarks[6].x * w, landmarks[6].y * h]) 
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

    # [2] 얼굴 높이 계산 및 측면 판별 (원본 유지)
    current_face_height = np.sqrt((nose_bridge[0] - chin[0])**2 + (nose_bridge[1] - chin[1])**2)
    eye_dist = np.sqrt((r_eye[0]-l_eye[0])**2 + (r_eye[1]-l_eye[1])**2)
    side_ratio = eye_dist / current_face_height
    is_profile = side_ratio < 0.50 
    
    # [3] 배율 설정 (원본 유지)
    target_face_height = h * 0.28
    base_scale = target_face_height / current_face_height
    scale = base_scale * (0.70 if is_profile else 1.0)
    
    # [4] 변환 행렬 생성 및 캔버스 크기 최적화 (수정)
    M = cv2.getRotationMatrix2D(tuple(nose_bridge), angle, scale)

    # 빈 공간 최소화를 위해 회전 후의 새로운 크기 계산
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # [5] 라인 고정 로직 (기존 비율 유지하되 이동 보정)
    # 현재 미간 위치를 변환 후 어디로 가는지 확인
    curr_bridge_trans = M @ np.array([nose_bridge[0], nose_bridge[1], 1])
    
    # 중심점 이동 보정: 캔버스 크기가 바뀌었으므로 (new_w, new_h) 기준으로 정렬
    M[0, 2] += (new_w * 0.5 - curr_bridge_trans[0])
    M[1, 2] += (new_h * 0.40 - curr_bridge_trans[1])
    
    # [6] 측면 전용 추가 수직 보정 (원본 유지)
    if is_profile:
        M[1, 2] -= (new_h * 0.05) 

    # [결과 생성] borderMode를 BORDER_REPLICATE로 변경하여 빈 공간 최소화
    aligned_img = cv2.warpAffine(
        img_array, 
        M, 
        (new_w, new_h), 
        borderMode=cv2.BORDER_REPLICATE, # 주변색 확장
        flags=cv2.INTER_LINEAR
    )
    
    return aligned_img
