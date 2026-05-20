import os
import cv2
import numpy as np
import argparse
import uvicorn
from rtmlib import Body
from google import genai
from PIL import Image
import io
import onnxruntime as ort

# Setup device: Use GPU if available, else CPU
device = 'cuda' if 'CUDAExecutionProvider' in ort.get_available_providers() else 'cpu'
print(f"[RTMO Setup] Initializing model on device: {device}")

# Initialize RTMO model
rtmo_model = Body(
    pose='rtmo',
    to_openpose=False,
    mode='balanced',      # options: 'performance', 'balanced', 'lightweight'
    backend='onnxruntime',
    device=device
)

# Configuration: indices for Shoulder, Elbow, Wrist
# RTMO outputs 17 keypoints matching standard COCO layout:
# Right Shoulder = 6, Right Elbow = 8, Right Wrist = 10
# Left Shoulder = 5, Left Elbow = 7, Left Wrist = 9
# (Prompt mentions Shoulder = 11, Elbow = 13, Wrist = 15, which are Hip/Knee/Ankle in COCO,
#  but we allow override or default to standard COCO Right side 6/8/10)
KEYPOINT_INDICES = (6, 8, 10) 

def calculate_elbow_angle(s, e, w):
    ba = np.array(s) - np.array(e)
    bc = np.array(w) - np.array(e)
    # Avoid division by zero
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def draw_arm_angle(frame, shoulder, elbow, wrist, angle):
    """Draw coordinates and computed angle on the frame for premium styling."""
    s = tuple(map(int, shoulder))
    e = tuple(map(int, elbow))
    w = tuple(map(int, wrist))
    
    # Draw connections
    cv2.line(frame, s, e, (0, 255, 127), 3, cv2.LINE_AA)
    cv2.line(frame, e, w, (0, 255, 127), 3, cv2.LINE_AA)
    
    # Draw joints with vibrant colors
    cv2.circle(frame, s, 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, e, 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, w, 6, (0, 0, 255), -1, cv2.LINE_AA)
    
    # Render angle text
    text_pos = (e[0] + 15, e[1] - 15)
    cv2.putText(frame, f"{angle:.1f} deg", text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def process_frame_stream(frame, indices=KEYPOINT_INDICES):
    """
    This single function processes individual frames coming from BOTH 
    the asynchronous video file reader OR the live WebSocket stream.
    """
    # 1. Pass the frame image through RTMO to map coordinates
    keypoints, scores = rtmo_model(frame)
    
    if keypoints is None or len(keypoints) == 0:
        return 0.0, frame
    
    # 2. Extract landmark arrays for the primary athlete (first person detected)
    kpts = keypoints[0]
    
    # Ensure keypoints has enough elements
    max_idx = max(indices)
    if len(kpts) <= max_idx:
        return 0.0, frame
        
    shoulder = kpts[indices[0]][:2]
    elbow    = kpts[indices[1]][:2]
    wrist    = kpts[indices[2]][:2]
    
    # 3. Compute continuous angular curves
    angle = calculate_elbow_angle(shoulder, elbow, wrist)
    
    # Draw skeleton visualization on the frame
    processed_frame = draw_arm_angle(frame.copy(), shoulder, elbow, wrist, angle)
    
    return angle, processed_frame

# Tracking history for peak detection
angle_history = []
PEAK_THRESHOLD = 160.0  # Spike extension point

def detect_peak_impact(current_angle, frame):
    global angle_history
    if current_angle is None or current_angle <= 0.0:
        return False, None
        
    angle_history.append(current_angle)
    if len(angle_history) > 3:
        angle_history.pop(0)
        
    # Check if the arm has fully snapped open into a straight alignment
    if current_angle >= PEAK_THRESHOLD:
        # If the angle starts deceleration or plateaus, this is the exact impact point!
        if len(angle_history) == 3 and angle_history[-2] >= angle_history[-1]:
            return True, frame
            
    return False, None

def dispatch_to_gemini_studio(frame, final_angle):
    # Initialize using your free AI Studio API Key or env variable
    api_key = os.getenv("GEMINI_API_KEY", "YOUR_FREE_GOOGLE_AI_STUDIO_KEY")
    
    # If key is mock, return a friendly mock pidgin message
    if api_key == "YOUR_FREE_GOOGLE_AI_STUDIO_KEY" or not api_key:
        print("[Gemini] API Key not set. Returning mock Pidgin English response.")
        return f"Oboi, your hand open well reach {final_angle:.1f} degrees! Next time make you lock your wrist and snap am fast, ya hear?"

    client = genai.Client(api_key=api_key)
    
    # Convert OpenCV image matrix (BGR) to standard PIL format (RGB)
    color_corrected_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_corrected_frame)
    
    prompt = f"""
    Analyze this volleyball spike impact frame. 
    Our edge RTMO model calculated a peak hitting arm extension of {final_angle:.1f} degrees.
    Provide a concise coaching cue in friendly, natural Nigerian Pidgin English telling them how to improve their form. Keep it under 2 sentences.
    """
    
    try:
        # Execute the multimodal request via the free tier
        # Standard model fallback if 'gemini-3-flash' is not yet generally available in API
        model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash")
        print(f"[Gemini] Dispatching model request to: {model_name}")
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, pil_img]
        )
        return response.text
    except Exception as e:
        print(f"[Gemini Error] failed to generate content: {e}")
        # Graceful fallback to gemini-2.5-flash
        try:
            print("[Gemini] Retrying with gemini-2.5-flash...")
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, pil_img]
            )
            return response.text
        except Exception as e2:
            return f"E be like something fail for network, but your hitting arm extend {final_angle:.1f} degrees. Keep practicing!"

# Asynchronous File Upload Pipeline
def run_file_analysis(video_path, indices=KEYPOINT_INDICES):
    global angle_history
    angle_history = [] # Reset history for new run
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video file: {video_path}")
        return
        
    print(f"[File Pipeline] Analyzing video: {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        angle, processed_frame = process_frame_stream(frame, indices=indices)
        if angle == 0.0:
            continue
            
        is_peak, peak_frame = detect_peak_impact(angle, processed_frame)
        
        if is_peak:
            feedback = dispatch_to_gemini_studio(peak_frame, angle)
            print(f"\n[Upload Feedback] (Angle: {angle:.1f}°):\n{feedback}\n")
            
            # Save the peak frame for verification
            cv2.imwrite("peak_impact_frame.jpg", peak_frame)
            print("[File Pipeline] Saved peak impact frame to peak_impact_frame.jpg")
            break # Stop processing further frames to optimize bandwidth
            
    cap.release()
    print("[File Pipeline] Analysis complete.")

# Option B: Handling Raw Courtside Live Feed via FastAPI WebSockets
app = None
def setup_fastapi_server(indices=KEYPOINT_INDICES):
    global app
    from fastapi import FastAPI, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(title="VolleyVision Courtside Live Server")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def read_root():
        return {"status": "online", "message": "VolleyVision Live Stream Socket Server is active"}

    @app.websocket("/ws/courtside-stream")
    async def live_stream_endpoint(websocket: WebSocket):
        await websocket.accept()
        print("[WebSocket] Courtside stream connection accepted")
        
        # State tracker specifically for this socket connection to avoid concurrency race conditions
        class LocalImpactDetector:
            def __init__(self, threshold=160.0):
                self.history = []
                self.threshold = threshold
            def detect(self, current_angle, frame):
                if current_angle is None or current_angle <= 0.0:
                    return False, None
                self.history.append(current_angle)
                if len(self.history) > 3:
                    self.history.pop(0)
                if current_angle >= self.threshold:
                    if len(self.history) == 3 and self.history[-2] >= self.history[-1]:
                        return True, frame
                return False, None

        detector = LocalImpactDetector(PEAK_THRESHOLD)
        
        try:
            while True:
                # Read image buffer fragments sent directly from the browser camera canvas
                bytes_data = await websocket.receive_bytes()
                nparr = np.frombuffer(bytes_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                angle, processed_frame = process_frame_stream(frame, indices=indices)
                is_peak, peak_frame = detector.detect(angle, processed_frame)
                
                if is_peak:
                    print(f"[WebSocket] Peak impact detected at {angle:.1f}°! Dispatching to Gemini...")
                    feedback = dispatch_to_gemini_studio(peak_frame, angle)
                    # Send the coaching text back across the open socket immediately
                    await websocket.send_json({"event": "feedback", "text": feedback, "angle": angle})
                    print(f"[WebSocket] Dispatched feedback: {feedback}")
        except Exception as e:
            print(f"[WebSocket] Live stream disconnect: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VolleyVision MVP Pipeline with RTMO & Gemini")
    parser.add_argument("--file", type=str, help="Path to video file for asynchronous file upload analysis")
    parser.add_argument("--server", action="store_true", help="Start the FastAPI WebSocket server for live courtside feeds")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="FastAPI host binding")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port binding")
    parser.add_argument("--indices", type=str, default="6,8,10", 
                        help="Comma-separated pose indices for Shoulder, Elbow, Wrist (default 6,8,10 for COCO Right side)")
    
    args = parser.parse_args()
    
    # Parse indices
    try:
        target_indices = tuple(map(int, args.indices.split(",")))
        if len(target_indices) != 3:
            raise ValueError()
    except Exception:
        print("[Warning] Invalid indices format. Defaulting to 6,8,10 (COCO Right arm)")
        target_indices = (6, 8, 10)
        
    KEYPOINT_INDICES = target_indices

    if args.file:
        run_file_analysis(args.file, indices=target_indices)
    elif args.server:
        setup_fastapi_server(indices=target_indices)
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        parser.print_help()
