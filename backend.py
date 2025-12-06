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

# Globals (set on startup)
app = FastAPI()
# Add permissive CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_frame_lock = threading.Lock()
_latest_frame = None  # stores most recent BGR numpy array
_stop_event = threading.Event()

model_lock = threading.Lock()
tokenizer = None
model = None
image_processor = None
device = None
base_prompt_template = None

# Fall-detection globals
fall_lock = threading.Lock()
_fall_status = {
    "trigger_frames": 0,
    "visible": 0,
    "z_range": 0.0,
    "z_range_text": "",
    "fall_detected": False,
    "last_updated": None,
}

class DescribeResponse(BaseModel):
    timestamp: str
    caption: str

class SearchResponse(BaseModel):
    timestamp: str
    result: str

class FallResponse(BaseModel):
    timestamp: Optional[str]
    fall_detected: bool
    trigger_frames: int
    visible: int
    z_range: float
    z_range_text: str

def _start_frame_reader(rtsp_url: Optional[str]):
    def reader():
        nonlocal rtsp_url
        cap = cv2.VideoCapture(rtsp_url or 0)
        if not cap.isOpened():
            print(f"[frame_reader] Failed to open stream: {rtsp_url}, trying device 0")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[frame_reader] Failed to open any capture device.")
                return
        print(f"[frame_reader] Capture started: {rtsp_url or 'device 0'}")
        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            with _frame_lock:
                # store BGR frame
                global _latest_frame
                _latest_frame = frame
        cap.release()
        print("[frame_reader] Capture stopped.")
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    return t

def _get_latest_pil():
    with _frame_lock:
        frame = _latest_frame.copy() if _latest_frame is not None else None
    if frame is None:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def _prepare_prompt(conv_mode: str, prompt_text: str):
    qs = prompt_text or "Describe the image."
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def _generate_from_prompt(prompt: str, pil_img: Image.Image, temperature: float, top_p: Optional[float], num_beams: int, max_new_tokens: int):
    # Tokenize prompt and prepare input ids
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    # Preprocess image(s)
    image_tensor = process_images([pil_img], image_processor, model.config)[0]
    if device.type in ("cuda", "mps"):
        images_input = image_tensor.unsqueeze(0).half().to(device)
    else:
        images_input = image_tensor.unsqueeze(0).to(device)

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

@app.get("/describe", response_model=DescribeResponse)
async def describe(conv_mode: str = "qwen_2", prompt: Optional[str] = None, temperature: float = 0.2,
                   top_p: Optional[float] = None, num_beams: int = 1, max_new_tokens: int = 256):
    pil = _get_latest_pil()
    if pil is None:
        raise HTTPException(status_code=503, detail="No frame available yet")
    # use prepared conv template if available
    prompt_text = _prepare_prompt(conv_mode, prompt or "Describe the image in a few sentences, focus on objects/people in the room, and not the room itself.")
    loop = asyncio.get_event_loop()
    caption = await loop.run_in_executor(None, _generate_from_prompt, prompt_text, pil, temperature, top_p, num_beams, max_new_tokens)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return DescribeResponse(timestamp=timestamp, caption=caption)

@app.get("/search", response_model=SearchResponse)
async def search(object: str = Query(..., description="Object to search for in the image"),
                 conv_mode: str = "qwen_2", temperature: float = 0.0,
                 top_p: Optional[float] = None, num_beams: int = 1, max_new_tokens: int = 128):
    pil = _get_latest_pil()
    if pil is None:
        raise HTTPException(status_code=503, detail="No frame available yet")
    # Create a question prompt that asks about the object explicitly
    question = f"Is there a {object} in the image? If yes, briefly describe where it is, specifically in relation to other objects. If no, answer 'No'."
    prompt_text = _prepare_prompt(conv_mode, question)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _generate_from_prompt, prompt_text, pil, temperature, top_p, num_beams, max_new_tokens)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return SearchResponse(timestamp=timestamp, result=result)

@app.get("/fall", response_model=FallResponse)
async def fall_status():
    with fall_lock:
        last = _fall_status.copy()
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

def _start_fall_detector(poll_interval: float = 0.1, visibility_threshold: float = 0.8, z_range_threshold: float = 0.4, trigger_limit: int = 60):
    """
    Background thread: periodically reads latest frame, runs MediaPipe Pose,
    computes visible landmarks, z_range, updates trigger_frames and fall_detected.
    """
    mp_pose = mp.solutions.pose

    def detector():
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while not _stop_event.is_set():
                pil = _get_latest_pil()
                if pil is None:
                    time.sleep(poll_interval)
                    continue
                # Actually get numpy array from PIL reliably:
                frame_np = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                # Now prepare for mediapipe (expects RGB)
                frame_for_mp = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

                results = pose.process(frame_for_mp)
                with fall_lock:
                    if not results.pose_landmarks:
                        # no landmarks
                        _fall_status["trigger_frames"] = 0
                        _fall_status["visible"] = 0
                        _fall_status["z_range"] = 0.0
                        _fall_status["z_range_text"] = ""
                        _fall_status["fall_detected"] = False
                        _fall_status["last_updated"] = time.time()
                    else:
                        lms = results.pose_landmarks.landmark
                        visible = len([lm for lm in lms if lm.visibility > visibility_threshold])
                        # use y coordinates similar to original main.py
                        y_coords = [lm.y for lm in lms]
                        z_range = max(y_coords) - min(y_coords) if y_coords else 0.0
                        z_range_text = f"Range: {round(z_range, 2)}; Trigger: {_fall_status['trigger_frames']}; Visible: {visible}"

                        if z_range < z_range_threshold:
                            _fall_status["trigger_frames"] += 1
                        else:
                            _fall_status["trigger_frames"] = 0

                        fall_detected = (_fall_status["trigger_frames"] > trigger_limit and visible > 15)
                        _fall_status.update({
                            "visible": visible,
                            "z_range": float(z_range),
                            "z_range_text": z_range_text,
                            "fall_detected": bool(fall_detected),
                            "last_updated": time.time(),
                        })
                time.sleep(poll_interval)

    import numpy as np  # local import to keep top-level imports minimal
    t = threading.Thread(target=detector, daemon=True)
    t.start()
    return t

def start_server_and_resources(args):
    global tokenizer, model, image_processor, device
    print("Loading model...")
    disable_torch_init()
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    device = torch.device(args.device)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name, device=str(device))
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded.")
    # start frame reader
    _start_frame_reader(args.rtsp_url)
    # start fall detector
    _start_fall_detector()
    # start uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTSP live captioning API server using LLaVA-style VLM")
    parser.add_argument("--rtsp-url", type=str, default=None, help="RTSP stream URL (if omitted, uses default camera device)")
    parser.add_argument("--model-path", type=str, default="./llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu", help="torch device (cpu, cuda, mps)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    args = parser.parse_args()

    try:
        start_server_and_resources(args)
    except KeyboardInterrupt:
        _stop_event.set()
        print("Server stopping...")
