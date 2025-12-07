"""
Sixth Sense Backend Server - AI-Powered Vision Monitoring API

This FastAPI server provides real-time video analysis capabilities including:
- Scene understanding and description using LLaVA vision-language model
- Object detection and spatial location identification
- Fall detection using MediaPipe pose estimation
- Real-time video frame processing from RTSP streams or local cameras

The server runs three concurrent background threads:
1. Frame reader: Continuously captures video frames from camera
2. Fall detector: Analyzes pose landmarks to detect potential falls
3. API server: Handles HTTP requests for scene analysis

Author: Sixth Sense Team
License: MIT
"""

import argparse
import threading
import time
import cv2
from PIL import Image
import torch
import asyncio
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional
import mediapipe as mp

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# ============================================================================
# Global State Variables
# ============================================================================

# FastAPI application instance
app = FastAPI()

# CORS middleware configuration - allows cross-origin requests from any domain
# This is necessary for the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Video frame state management
_frame_lock = threading.Lock()  # Thread-safe access to frame buffer
_latest_frame = None  # Stores most recent BGR numpy array from camera
_stop_event = threading.Event()  # Signal to stop background threads

# AI model state management
model_lock = threading.Lock()  # Thread-safe access to model
tokenizer = None  # LLaVA tokenizer
model = None  # LLaVA vision-language model
image_processor = None  # Image preprocessing pipeline
device = None  # PyTorch device (CPU, CUDA, or MPS)
base_prompt_template = None  # Template for model prompts

# Fall detection state management
fall_lock = threading.Lock()  # Thread-safe access to fall detection state
_fall_status = {
    "trigger_frames": 0,  # Number of consecutive frames suggesting a fall
    "visible": 0,  # Number of visible pose landmarks
    "z_range": 0.0,  # Vertical range of pose (low values suggest horizontal position)
    "z_range_text": "",  # Human-readable fall detection metrics
    "fall_detected": False,  # Whether a fall is currently detected
    "last_updated": None,  # Timestamp of last update
}

# ============================================================================
# API Response Models
# ============================================================================

class DescribeResponse(BaseModel):
    """Response model for scene description endpoint"""
    timestamp: str
    caption: str

class SearchResponse(BaseModel):
    """Response model for object search endpoint"""
    timestamp: str
    result: str

class FallResponse(BaseModel):
    """Response model for fall detection status endpoint"""
    timestamp: Optional[str]
    fall_detected: bool
    trigger_frames: int
    visible: int
    z_range: float
    z_range_text: str

# ============================================================================
# Background Thread: Video Frame Reader
# ============================================================================

def _start_frame_reader(rtsp_url: Optional[str]):
    """
    Start a background thread that continuously reads frames from the video source.
    
    This thread runs indefinitely, capturing frames from either an RTSP stream
    or a local camera device, and stores the latest frame in shared memory.
    
    Args:
        rtsp_url: RTSP stream URL, or None to use default camera (device 0)
        
    Returns:
        Thread: The started background thread
    """
    def reader():
        nonlocal rtsp_url
        # Attempt to open the video source
        cap = cv2.VideoCapture(rtsp_url or 0)
        if not cap.isOpened():
            print(f"[frame_reader] Failed to open stream: {rtsp_url}, trying device 0")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[frame_reader] Failed to open any capture device.")
                return
        print(f"[frame_reader] Capture started: {rtsp_url or 'device 0'}")
        
        # Continuous frame capture loop
        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            # Thread-safe update of latest frame
            with _frame_lock:
                global _latest_frame
                _latest_frame = frame
        
        # Cleanup on thread termination
        cap.release()
        print("[frame_reader] Capture stopped.")
    
    # Start the reader thread as a daemon (exits when main program exits)
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    return t

# ============================================================================
# Helper Functions
# ============================================================================

def _get_latest_pil():
    """
    Retrieve the latest video frame as a PIL Image.
    
    Thread-safe retrieval of the most recent frame from the video stream,
    converting from OpenCV BGR format to PIL RGB format.
    
    Returns:
        PIL.Image: The latest frame in RGB format, or None if no frame available
    """
    with _frame_lock:
        frame = _latest_frame.copy() if _latest_frame is not None else None
    if frame is None:
        return None
    if frame is None:
        return None
    # Convert from BGR (OpenCV format) to RGB (PIL format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def _prepare_prompt(conv_mode: str, prompt_text: str):
    """
    Prepare a prompt for the LLaVA vision-language model.
    
    Formats the user's text prompt according to LLaVA's conversation template,
    including special tokens for image understanding.
    
    Args:
        conv_mode: Conversation mode/template to use (e.g., "qwen_2")
        prompt_text: The user's question or instruction about the image
        
    Returns:
        str: Formatted prompt ready for model input
    """
    qs = prompt_text or "Describe the image."
    
    # Add image tokens if model supports them
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
    # Use conversation template to format prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def _generate_from_prompt(prompt: str, pil_img: Image.Image, temperature: float, 
                         top_p: Optional[float], num_beams: int, max_new_tokens: int):
    """
    Generate a text response from the vision-language model.
    
    Takes an image and text prompt, processes them through LLaVA model,
    and generates a natural language response describing or answering about the image.
    
    Args:
        prompt: Formatted text prompt for the model
        pil_img: PIL Image to analyze
        temperature: Sampling temperature (0 = deterministic, higher = more random)
        top_p: Nucleus sampling parameter (probability threshold)
        num_beams: Number of beams for beam search
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        str: Generated text response from the model
    """
    # Tokenize the text prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    
    # Preprocess the image for model input
    image_tensor = process_images([pil_img], image_processor, model.config)[0]
    
    # Move to appropriate device and convert to half precision if supported
    if device.type in ("cuda", "mps"):
        images_input = image_tensor.unsqueeze(0).half().to(device)
    else:
        images_input = image_tensor.unsqueeze(0).to(device)

    # Generate response using the model (thread-safe)
    with model_lock:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_input,
                image_sizes=[pil_img.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
            caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return caption

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/describe", response_model=DescribeResponse)
async def describe(conv_mode: str = "qwen_2", prompt: Optional[str] = None, 
                   temperature: float = 0.2, top_p: Optional[float] = None, 
                   num_beams: int = 1, max_new_tokens: int = 256):
    """
    Describe the current scene from the video feed.
    
    Uses the LLaVA vision-language model to generate a natural language
    description of what's currently visible in the camera frame.
    
    Args:
        conv_mode: Conversation template to use
        prompt: Custom prompt (default: asks to describe image focusing on objects/people)
        temperature: Sampling temperature for text generation
        top_p: Nucleus sampling threshold
        num_beams: Beam search parameter
        max_new_tokens: Maximum length of generated description
        
    Returns:
        DescribeResponse: Timestamp and generated scene description
        
    Raises:
        HTTPException: 503 if no video frame is available
    """
    pil = _get_latest_pil()
    if pil is None:
        raise HTTPException(status_code=503, detail="No frame available yet")
    
    # Prepare prompt focusing on objects and people rather than room layout
    prompt_text = _prepare_prompt(
        conv_mode, 
        prompt or "Describe the image in a few sentences, focus on objects/people in the room, and not the room itself."
    )
    
    # Run model inference in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    caption = await loop.run_in_executor(None, _generate_from_prompt, prompt_text, pil, temperature, top_p, num_beams, max_new_tokens)
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return DescribeResponse(timestamp=timestamp, caption=caption)

@app.get("/search", response_model=SearchResponse)
async def search(object: str = Query(..., description="Object to search for in the image"),
                 conv_mode: str = "qwen_2", temperature: float = 0.0,
                 top_p: Optional[float] = None, num_beams: int = 1, max_new_tokens: int = 128):
    """
    Search for a specific object in the current video frame.
    
    Uses AI to determine if the specified object is present and describe its
    location relative to other objects in the scene.
    
    Args:
        object: Name of the object to search for (required)
        conv_mode: Conversation template to use
        temperature: Sampling temperature (0 = deterministic, best for search)
        top_p: Nucleus sampling threshold
        num_beams: Beam search parameter
        max_new_tokens: Maximum length of response
        
    Returns:
        SearchResponse: Timestamp and search result (location or "No")
        
    Raises:
        HTTPException: 503 if no video frame is available
    """
    pil = _get_latest_pil()
    if pil is None:
        raise HTTPException(status_code=503, detail="No frame available yet")
    
    # Create specific question about object location
    question = f"Is there a {object} in the image? If yes, briefly describe where it is, specifically in relation to other objects. If no, answer 'No'."
    prompt_text = _prepare_prompt(conv_mode, question)
    
    # Run model inference
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _generate_from_prompt, prompt_text, pil, temperature, top_p, num_beams, max_new_tokens)
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return SearchResponse(timestamp=timestamp, result=result)

@app.get("/fall", response_model=FallResponse)
async def fall_status():
    """
    Get the current fall detection status.
    
    Returns real-time information about whether a fall has been detected,
    along with diagnostic metrics about pose tracking.
    
    Returns:
        FallResponse: Fall detection status and metrics
    """
    with fall_lock:
        last = _fall_status.copy()
    
    # Format timestamp if available
    if last["last_updated"] is None:
        timestamp = None
    else:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last["last_updated"]))
    
    return FallResponse(
        timestamp=timestamp,
        fall_detected=last["fall_detected"],
        trigger_frames=last["trigger_frames"],
        visible=last["visible"],
        z_range=last["z_range"],
        z_range_text=last["z_range_text"],
    )

# ============================================================================
# Background Thread: Fall Detection
# ============================================================================

def _start_fall_detector(poll_interval: float = 0.1, visibility_threshold: float = 0.8, 
                         z_range_threshold: float = 0.4, trigger_limit: int = 60):
    """
    Start a background thread for real-time fall detection.
    
    Uses MediaPipe Pose estimation to track body landmarks and detect when
    a person appears to be in a horizontal position (potential fall).
    
    Algorithm:
    1. Detect pose landmarks in each frame
    2. Calculate vertical range (y-axis) of pose
    3. If range is small (person horizontal), increment trigger counter
    4. If trigger counter exceeds threshold and enough landmarks visible, flag fall
    
    Args:
        poll_interval: How often to check for falls (seconds)
        visibility_threshold: Minimum landmark visibility to count as visible
        z_range_threshold: Maximum vertical range to consider as potential fall
        trigger_limit: Number of consecutive frames needed to confirm fall
        
    Returns:
        Thread: The started background thread
    """
    mp_pose = mp.solutions.pose

    def detector():
        # Initialize MediaPipe Pose detection
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, 
                         min_tracking_confidence=0.5) as pose:
            while not _stop_event.is_set():
                # Get latest frame
                pil = _get_latest_pil()
                if pil is None:
                    time.sleep(poll_interval)
                    continue
                
                # Convert PIL to numpy array for MediaPipe
                import numpy as np
                frame_np = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                frame_for_mp = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

                # Run pose detection
                results = pose.process(frame_for_mp)
                
                with fall_lock:
                    if not results.pose_landmarks:
                        # No person detected - reset fall detection
                        _fall_status["trigger_frames"] = 0
                        _fall_status["visible"] = 0
                        _fall_status["z_range"] = 0.0
                        _fall_status["z_range_text"] = ""
                        _fall_status["fall_detected"] = False
                        _fall_status["last_updated"] = time.time()
                    else:
                        # Analyze pose landmarks
                        lms = results.pose_landmarks.landmark
                        
                        # Count highly visible landmarks
                        visible = len([lm for lm in lms if lm.visibility > visibility_threshold])
                        
                        # Calculate vertical range (y-axis spread) of pose
                        # Low range suggests person is horizontal (potential fall)
                        y_coords = [lm.y for lm in lms]
                        z_range = max(y_coords) - min(y_coords) if y_coords else 0.0
                        z_range_text = f"Range: {round(z_range, 2)}; Trigger: {_fall_status['trigger_frames']}; Visible: {visible}"

                        # Update trigger counter
                        if z_range < z_range_threshold:
                            _fall_status["trigger_frames"] += 1
                        else:
                            _fall_status["trigger_frames"] = 0

                        # Determine if fall is detected
                        # Requires: enough consecutive frames + enough visible landmarks
                        fall_detected = (_fall_status["trigger_frames"] > trigger_limit and visible > 15)
                        
                        _fall_status.update({
                            "visible": visible,
                            "z_range": float(z_range),
                            "z_range_text": z_range_text,
                            "fall_detected": bool(fall_detected),
                            "last_updated": time.time(),
                        })
                
                time.sleep(poll_interval)

    # Start detector thread as daemon
    t = threading.Thread(target=detector, daemon=True)
    t.start()
    return t

# ============================================================================
# Server Initialization
# ============================================================================

def start_server_and_resources(args):
    """
    Initialize all server components and start the API server.
    
    Loads the AI model, starts background threads for frame capture and
    fall detection, then launches the FastAPI server.
    
    Args:
        args: Command-line arguments containing model paths and server config
    """
    global tokenizer, model, image_processor, device
    
    # Load the LLaVA vision-language model
    print("Loading model...")
    disable_torch_init()
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    device = torch.device(args.device)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name, device=str(device)
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded.")
    
    # Start background services
    _start_frame_reader(args.rtsp_url)
    _start_fall_detector()
    
    # Start the API server
    uvicorn.run(app, host=args.host, port=args.port)

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sixth Sense Backend - AI-Powered Vision Monitoring API"
    )
    parser.add_argument("--rtsp-url", type=str, default=None, 
                       help="RTSP stream URL (if omitted, uses default camera device)")
    parser.add_argument("--model-path", type=str, default="./llava-v1.5-13b",
                       help="Path to LLaVA model")
    parser.add_argument("--model-base", type=str, default=None,
                       help="Base model path (optional)")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="PyTorch device: cpu, cuda, or mps")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="API server host address")
    parser.add_argument("--port", type=int, default=8000, 
                       help="API server port")
    args = parser.parse_args()

    try:
        start_server_and_resources(args)
    except KeyboardInterrupt:
        _stop_event.set()
        print("\n[Server] Shutting down gracefully...")
