import cv2
import mediapipe as mp
import numpy as np
import os
import time
import urllib.request

# --- MEDIAPIPE SETUP (Hands) ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- DOWNLOAD MODELS IF MISSING ---
def ensure_models(script_dir):
    assets_dir = os.path.join(script_dir, 'assets')
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        
    models = {
        "face_detection_yunet.onnx": "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "face_recognition_sface.onnx": "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"
    }
    
    for name, url in models.items():
        path = os.path.join(assets_dir, name)
        if not os.path.exists(path):
            print(f"Downloading {name}... (~30MB total, please wait)")
            urllib.request.urlretrieve(url, path)
            print(f"Downloaded {name}")
    return os.path.join(assets_dir, "face_detection_yunet.onnx"), os.path.join(assets_dir, "face_recognition_sface.onnx")

def load_signatures(signatures_dir):
    signatures = {}
    if not os.path.exists(signatures_dir):
        os.makedirs(signatures_dir)
    
    for filename in os.listdir(signatures_dir):
        if filename.endswith(".npy"):
            name = filename[:-4]
            path = os.path.join(signatures_dir, filename)
            signatures[name] = np.load(path)
            print(f"- Loaded signature for: {name}")
    return signatures

def main():
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hand_model_path = os.path.join(script_dir, 'hand_landmarker.task')
    output_file = os.path.join(script_dir, 'wave_log.txt')
    signatures_dir = os.path.join(script_dir, 'assets', 'signatures')

    # Ensure Face Recognition models are available
    det_model_path, rec_model_path = ensure_models(script_dir)

    # Initialize Face Detector (YuNet) & Recognizer (SFace)
    detector = cv2.FaceDetectorYN.create(det_model_path, "", (320, 320))
    recognizer = cv2.FaceRecognizerSF.create(rec_model_path, "")

    # Load all recorded face signatures
    print("--- Loading All Signatures ---")
    signatures = load_signatures(signatures_dir)

    # MediaPipe Hand Landmarker Configuration
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Variables for state
    history_x = []
    MAX_HISTORY = 20
    wave_cooldown = 0
    namaste_cooldown = 0
    recognition_cooldowns = {} # name: time mapping
    COOLDOWN_TIME = 2.0 
    
    cap = cv2.VideoCapture(0)
    frame_idx = 0
    
    print(f"--- Multi-Person AI Initialized ---")
    print(f"Shortcuts: 's' to Save Face (Enroll in Terminal), 'q' to Quit")

    with HandLandmarker.create_from_options(options) as hand_landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # --- FACE DETECTION & MULTI-RECOGNITION ---
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            
            hud_status = "Scanning for identity..."
            anyone_recognized = False

            if faces is not None:
                for face in faces:
                    coords = face[:4].astype(int)
                    # Standard face box
                    cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (255, 255, 255), 1)
                    
                    best_match_name = "Unknown"
                    max_score = 0.4 # Similarity Threshold for SFace (0.0 to 1.0)
                    
                    if signatures:
                        face_aligned = recognizer.alignCrop(frame, face)
                        face_feature = recognizer.feature(face_aligned)
                        
                        for name, known_feature in signatures.items():
                            score = recognizer.match(known_feature, face_feature, cv2.FaceRecognizerSF_FR_COSINE)
                            if score > max_score:
                                max_score = score
                                best_match_name = name
                    
                    if best_match_name != "Unknown":
                        anyone_recognized = True
                        hud_status = f"Hi, {best_match_name}!"
                        # Green box for recognized person
                        cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"{best_match_name}", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Individual Log with Identity-Specific Cooldown
                        current_time = time.time()
                        if current_time - recognition_cooldowns.get(best_match_name, 0) > 10.0:
                            with open(output_file, 'a') as f:
                                f.write(f"{best_match_name} Recognized - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            print(f"Logged {best_match_name} recognition!")
                            recognition_cooldowns[best_match_name] = current_time
                    else:
                        cv2.putText(frame, "Unknown User", (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # --- HAND GESTURES (MediaPipe) ---
            frame_timestamp_ms = int(time.time() * 1000)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            hand_results = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
            color = (0, 255, 0) if anyone_recognized else (0, 255, 255)

            if hand_results.hand_landmarks:
                num_hands = len(hand_results.hand_landmarks)
                for hand_lms in hand_results.hand_landmarks:
                    palm = hand_lms[9]
                    cv2.circle(frame, (int(palm.x * w), int(palm.y * h)), 8, color, -1)

                # NAMASTE
                if num_hands == 2:
                    h1, h2 = hand_results.hand_landmarks[0][9], hand_results.hand_landmarks[1][9]
                    dist = np.sqrt((h1.x - h2.x)**2 + (h1.y - h2.y)**2)
                    if dist < 0.12:
                        current_time = time.time()
                        if current_time - namaste_cooldown > COOLDOWN_TIME:
                            hud_status = "🙏 NAMASTE LOGGED!"
                            namaste_cooldown = current_time
                            with open(output_file, 'a') as f:
                                f.write(f"namste - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            print("Logged 'namste'!")

                # WAVE
                palm = hand_results.hand_landmarks[0][9]
                history_x.append(palm.x)
                if len(history_x) > MAX_HISTORY: history_x.pop(0)
                if len(history_x) >= MAX_HISTORY:
                    dx = np.diff(history_x)
                    if (np.max(history_x) - np.min(history_x)) > 0.2 and np.sum(np.diff(np.sign(dx)) != 0) >= 2:
                        current_time = time.time()
                        if current_time - wave_cooldown > COOLDOWN_TIME:
                            hud_status = "👋 HI LOGGED!"
                            wave_cooldown = current_time
                            with open(output_file, 'a') as f:
                                f.write(f"hi - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            print("Logged 'hi'!")

            # --- UI HUD ---
            cv2.rectangle(frame, (0, 0), (w, 50), (10, 10, 10), -1)
            cv2.putText(frame, hud_status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "Press 's' to Save Face | 'q' to Quit", (w-350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            
            if time.time() - wave_cooldown < 1.0: cv2.putText(frame, "HI!", (w//2-50, h//2), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 255, 0), 5)
            if time.time() - namaste_cooldown < 1.0: cv2.putText(frame, "NAMASTE!", (w//2-180, h//2), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 255), 5)

            cv2.imshow("Multi-Person Interaction", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            
            # --- MULTI-PERSON ENROLLMENT ---
            if key == ord('s'):
                if faces is not None:
                    # Highlight enrollment in console
                    print("\n" + "="*30)
                    print("FACIAL IDENTITY ENROLLMENT")
                    print("="*30)
                    name = input("Type the name for this face: ").strip()
                    if not name: 
                        name = f"User_{frame_idx}"
                    
                    face_aligned = recognizer.alignCrop(frame, faces[0])
                    feature = recognizer.feature(face_aligned)
                    
                    path = os.path.join(signatures_dir, f"{name}.npy")
                    np.save(path, feature)
                    print(f"IDENTITY SAVED AS: {name}")
                    print("="*30 + "\n")
                    
                    # Reload all signatures
                    signatures = load_signatures(signatures_dir)
                    
                    # Visual feedback on camera
                    cv2.rectangle(frame, (0,0), (w, h), (0, 255, 0), 10)
                    cv2.putText(frame, f"ENROLLED: {name}", (w//2-200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                    cv2.imshow("Multi-Person Interaction", frame)
                    cv2.waitKey(1000)
                else:
                    print("!!! No face detected to save. Look at the camera.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
