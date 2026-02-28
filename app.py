def align_precise_line_lock(img_array):
    if img_array is None: return None
    h, w, _ = img_array.shape
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if not results or not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # [1] 정밀 포인트 추출 (기존 코드 그대로)
    brow_y = (landmarks[105].y + landmarks[334].y) / 2 * h
    nose_bridge = np.array([landmarks[6].x * w, landmarks[6].y * h]) 
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

    # [2] 얼굴 높이 계산 (기존 코드 그대로)
    current_face_height = np.sqrt((nose_bridge[0] - chin[0])**2 + (nose_bridge[1] - chin[1])**2)
    eye_dist = np.sqrt((r_eye[0]-l_eye[0])**2 + (r_eye[1]-l_eye[1])**2)
    side_ratio = eye_dist / current_face_height
    is_profile = side_ratio < 0.50 
    
    # [3] 배율 설정 (기존 코드 그대로)
    target_face_height = h * 0.28
    base_scale = target_face_height / current_face_height
    scale = base_scale * (0.70 if is_profile else 1.0)
    
    # [4] 변환 행렬 생성 (기존 코드 그대로)
    M = cv2.getRotationMatrix2D(tuple(nose_bridge), angle, scale)
    
    # [5] 라인 고정 로직 (기존 코드 그대로)
    target_brow_y = h * 0.35
    target_chin_y = h * 0.61 if is_profile else h * 0.65
    curr_bridge_trans = M @ np.array([nose_bridge[0], nose_bridge[1], 1])
    
    M[0, 2] += (w * 0.5 - curr_bridge_trans[0])
    M[1, 2] += (h * 0.40 - curr_bridge_trans[1])
    
    # [6] 측면 전용 추가 수직 보정 (기존 코드 그대로)
    if is_profile:
        M[1, 2] -= (h * 0.05)

    # --- 여기서부터 수정: 빈 공간 최소화 및 주변색 확장 ---
    
    # 1. 회전/축소된 이미지의 실제 범위를 계산하여 캔버스 크기 결정
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    rect_trans = cv2.transform(np.array([rect]), M)[0]
    
    # 변환된 이미지의 바운딩 박스(최소/최대 좌표) 추출
    min_x, min_y = np.min(rect_trans, axis=0)
    max_x, max_y = np.max(rect_trans, axis=0)
    
    # 잘리는 공간이 없도록 새로운 가로/세로 크기 설정
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))
    
    # 2. 이미지가 캔버스 밖으로 나가지 않도록 행렬 M을 이동 보정
    # (min_x, min_y가 0보다 작으면 그만큼 이미지를 안으로 밀어 넣음)
    M[0, 2] -= min_x
    M[1, 2] -= min_y

    # 3. 주변 색을 늘리는 BORDER_REPLICATE 적용
    aligned_img = cv2.warpAffine(
        img_array, 
        M, 
        (new_w, new_h), 
        borderMode=cv2.BORDER_REPLICATE, 
        flags=cv2.INTER_LINEAR
    )
    # ----------------------------------------------
    
    return aligned_img
