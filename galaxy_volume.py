import cv2
import mediapipe as mp
import numpy as np
import math
import os
import urllib.request
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --- MODERN MEDIAPIPE TASKS SETUP ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- ADVANCED HOLOGRAPHIC SYSTEM ---
class HologramParticle:
    def __init__(self, index):
        self.index = index
        self.reset()
        
    def reset(self):
        self.angle_offset = np.random.uniform(0, 2 * math.pi)
        # Store as relative normalized radius (0 to 1.0)
        self.rel_radius = np.random.uniform(0.1, 1.0)
        self.arm_offset = (int(self.rel_radius * 4)) * (math.pi / 2)
        
        self.z = np.random.uniform(-30, 30)
        self.size = np.random.randint(1, 3)
        self.speed = 0.02 + (1.0 - self.rel_radius) * 0.03
        
        # Cyber Colors
        self.color = (255, 255, 100) if self.index % 3 == 0 else (255, 100, 255)
        if self.index % 10 == 0: self.color = (255, 255, 255)

    def get_3d_pos(self, time_val, intensity, hand_scale_factor):
        curr_angle = self.angle_offset + self.arm_offset + (time_val * self.speed) * (1 + intensity)
        # Use hand_scale_factor to shrink/grow the galaxy
        actual_radius = self.rel_radius * 160 * hand_scale_factor
        x = actual_radius * math.cos(curr_angle)
        y = actual_radius * math.sin(curr_angle)
        return x, y, self.z

def project_3d(x, y, z, cx, cy, f=500):
    z_eye = z + 500
    scale = f / z_eye
    return int(cx + x * scale), int(cy + y * scale), scale

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'hand_landmarker.task')

    # Audio Init
    devices = AudioUtilities.GetSpeakers()
    try: volume = devices.EndpointVolume
    except AttributeError:
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
    vol_range = volume.GetVolumeRange()
    min_vol, max_vol = vol_range[0], vol_range[1]

    # MediaPipe Init
    landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO, num_hands=1
    ))

    particles = [HologramParticle(i) for i in range(45)]
    cap, smooth_vol, frame_idx, start_time = cv2.VideoCapture(0), 50, 0, time.time()
    
    # NEW: Scale smoothing
    smooth_hand_scale = 0.5 

    print("--- Holographic AI Volume Running ---")

    with landmarker as hand_tracker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame = cv2.addWeighted(frame, 0.6, np.zeros(frame.shape, frame.dtype), 0, 0)
            
            curr_time = time.time() - start_time
            frame_idx += 1
            results = hand_tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), frame_idx * 33)
            
            palm_center, intensity = None, smooth_vol / 100.0

            if results.hand_landmarks:
                for hand_lms in results.hand_landmarks:
                    wrist, mcp = hand_lms[0], hand_lms[9]
                    palm_center = (int(mcp.x * w), int(mcp.y * h))
                    
                    # NEW: DEPTH SENSING
                    # Distance between wrist and knuckle is a good depth indicator
                    dist_raw = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)
                    # Normalize: hand far is ~0.1, hand very close is ~0.4. Let's map 0.15 as base.
                    hand_scale = np.clip(dist_raw / 0.15, 0.4, 1.8)
                    smooth_hand_scale = 0.8 * smooth_hand_scale + 0.2 * hand_scale
                    
                    angle = math.degrees(math.atan2(mcp.y - wrist.y, mcp.x - wrist.x))
                    vol_percent = np.clip(np.interp(angle, [-135, -45], [0, 100]), 0, 100)
                    smooth_vol = 0.8 * smooth_vol + 0.2 * vol_percent
                    volume.SetMasterVolumeLevel(np.interp(smooth_vol, [0, 100], [min_vol, max_vol]), None)

            if palm_center:
                glow_layer = np.zeros_like(frame)
                proj = []
                for p in particles:
                    x, y, z = p.get_3d_pos(curr_time, intensity, smooth_hand_scale)
                    sx, sy, scale = project_3d(x, y, z, palm_center[0], palm_center[1])
                    proj.append({'x': sx, 'y': sy, 'z': z, 'col': p.color, 'scale': scale})
                
                # connections
                for i in range(len(proj)):
                    distances = sorted([( (proj[i]['x']-proj[j]['x'])**2 + (proj[i]['y']-proj[j]['y'])**2, j) for j in range(len(proj)) if i!=j])
                    for d, idx in distances[:2]:
                        # Max connection dist also scales with hand depth!
                        if d < (100 * smooth_hand_scale)**2:
                            cv2.line(glow_layer, (proj[i]['x'], proj[i]['y']), (proj[idx]['x'], proj[idx]['y']), (180, 180, 100), 1)

                for p in proj:
                    cv2.circle(glow_layer, (p['x'], p['y']), int(2 * p['scale']), p['col'], -1)

                # Concentric HUD Rings (Iron Man Style) - Scaled with depth
                for r, s in [(140, 0.5), (155, -0.3)]:
                    scaled_r = int(r * smooth_hand_scale)
                    for a in range(0, 360, 20):
                        ra = math.radians(a + math.degrees(curr_time * s))
                        cv2.circle(glow_layer, (int(palm_center[0] + scaled_r * math.cos(ra)), int(palm_center[1] + scaled_r * math.sin(ra))), 1, (255, 255, 255), -1)

                # Volume HUD - Scaled with depth
                v_r = int(170 * smooth_hand_scale)
                cv2.ellipse(glow_layer, palm_center, (v_r, v_r), 0, -135, -45, (40, 40, 40), 1)
                cv2.ellipse(glow_layer, palm_center, (v_r, v_r), 0, -135, int(np.interp(smooth_vol, [0, 100], [-135, -45])), (0, 255, 255), 2)
                
                cv2.putText(frame, f"VOL: {int(smooth_vol)}%", (palm_center[0]-int(35*smooth_hand_scale), palm_center[1]-v_r-20), 
                            cv2.FONT_HERSHEY_PLAIN, max(0.5, 1.2 * smooth_hand_scale), (255, 255, 255), 1)

                frame = cv2.add(frame, cv2.GaussianBlur(glow_layer, (7, 7), 0))
                frame = cv2.add(frame, glow_layer)

            cv2.imshow("Advanced AI HUD", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
