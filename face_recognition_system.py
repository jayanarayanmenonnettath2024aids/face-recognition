#!/usr/bin/env python3
"""
face_recognition_system.py

Single-file face recognition system:
- Uses OpenCV DNN face detector (with auto-download of model files) or HaarCascade fallback
- Uses an ONNX embedding model (ArcFace-like). You must provide path to an ONNX model (see notes)
- Stores embeddings in SQLite and an Annoy index for fast lookup
- CLI: add, recognize, list, exportnoe
Date: 2025-10-18
"""

import os
import sys
import sqlite3
import time
import argparse
import struct
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
import cv2
import onnxruntime as ort
import requests
from annoy import AnnoyIndex

# -----------------------------
# Configuration - change here
# -----------------------------
DB_PATH = "faces.db"
ANNOY_INDEX_PATH = "faces.ann"
ANNOY_TREE_COUNT = 10         # rebuild trees when new faces added (small DB -> small value is OK)
EMBEDDING_DIM = None         # will be inferred from ONNX model output
SIMILARITY_THRESHOLD = 0.60  # cosine distance threshold (60% match required - balanced accuracy)
MIN_FACE_SIZE = 80           # minimum face width/height in pixels to process
TEMPORAL_FRAMES = 5          # number of consecutive frames to verify recognition
MIN_FACE_AREA_RATIO = 0.02   # minimum face area as ratio of frame area
FACE_DETECTOR_DNN_PROTOTXT = "deploy.prototxt"
FACE_DETECTOR_DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
# URLs (auto-download) for OpenCV DNN face detector files (from OpenCV's GitHub)
DNN_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
# ONNX model path - you must provide an ONNX face-embedding model suited for ArcFace/FaceNet.
# Recommended: a lightweight ArcFace-Mobile or MobileFaceNet ONNX model (112x112 input). Put path or URL here.
DEFAULT_ONNX_PATH = "arcface_mobilenetv2.onnx"

# -----------------------------
# Utilities: DB & Annoy
# -----------------------------
def create_db(conn):
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        metadata TEXT,
        embedding BLOB NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    conn.commit()

def embedding_to_bytes(vec: np.ndarray) -> bytes:
    # store as float32 bytes
    return vec.astype(np.float32).tobytes()

def bytes_to_embedding(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)

def add_face_to_db(conn, name, embedding, metadata=None):
    c = conn.cursor()
    meta_json = json.dumps(metadata) if metadata else None
    c.execute("INSERT INTO faces (name, metadata, embedding, created_at) VALUES (?, ?, ?, ?)",
              (name, meta_json, embedding_to_bytes(embedding), datetime.utcnow().isoformat()))
    conn.commit()
    return c.lastrowid

def load_all_embeddings(conn):
    c = conn.cursor()
    c.execute("SELECT id, name, embedding FROM faces ORDER BY id")
    rows = c.fetchall()
    ids, names, embeddings = [], [], []
    for r in rows:
        ids.append(r[0]); names.append(r[1]); embeddings.append(bytes_to_embedding(r[2]))
    return ids, names, embeddings

# -----------------------------
# Auto-download DNN face detector
# -----------------------------
def ensure_dnn_detector_files():
    if os.path.exists(FACE_DETECTOR_DNN_PROTOTXT) and os.path.exists(FACE_DETECTOR_DNN_MODEL):
        return True
    print("[*] DNN face detector files missing. Attempting to download (small files)...")
    try:
        r = requests.get(DNN_PROTO_URL, timeout=15)
        r.raise_for_status()
        with open(FACE_DETECTOR_DNN_PROTOTXT, "wb") as f:
            f.write(r.content)
        r = requests.get(DNN_MODEL_URL, timeout=30)
        r.raise_for_status()
        with open(FACE_DETECTOR_DNN_MODEL, "wb") as f:
            f.write(r.content)
        print("[+] DNN detector files downloaded.")
        return True
    except Exception as e:
        print("[!] Could not download DNN files:", e)
        print("[!] Will fallback to HaarCascade detector.")
        return False

# -----------------------------
# Face Detector (OpenCV DNN with Haar fallback)
# -----------------------------
class FaceDetector:
    def __init__(self, conf_threshold=0.75):
        self.conf_threshold = conf_threshold
        self.net = None
        self.haar = None
        if ensure_dnn_detector_files():
            try:
                self.net = cv2.dnn.readNetFromCaffe(FACE_DETECTOR_DNN_PROTOTXT, FACE_DETECTOR_DNN_MODEL)
                # Prefer OpenVINO / CUDA if available (optional)
            except Exception as e:
                print("[!] Failed to load DNN detector:", e)
                self.net = None
        if self.net is None:
            # HaarCascade fallback (fast, lower accuracy)
            casc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar = cv2.CascadeClassifier(casc_path)
            if self.haar.empty():
                raise RuntimeError("Failed to load Haar Cascade face detector.")
            print("[*] Using Haar Cascade face detector (fallback).")

    def detect(self, frame):
        """Return list of boxes [x1,y1,x2,y2] in pixel coordinates with size filtering"""
        h, w = frame.shape[:2]
        frame_area = h * w
        boxes = []
        if self.net:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf > self.conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    # clip
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
                    
                    # Filter by minimum size and area ratio
                    face_w, face_h = x2 - x1, y2 - y1
                    face_area = face_w * face_h
                    if face_w >= MIN_FACE_SIZE and face_h >= MIN_FACE_SIZE:
                        if face_area / frame_area >= MIN_FACE_AREA_RATIO:
                            boxes.append([x1, y1, x2, y2])
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
            for (x, y, rw, rh) in rects:
                face_area = rw * rh
                if face_area / frame_area >= MIN_FACE_AREA_RATIO:
                    boxes.append([x, y, x + rw, y + rh])
        return boxes

# -----------------------------
# ONNX Embedding Model wrapper
# -----------------------------
class ONNXEmbedder:
    def __init__(self, onnx_path):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")
        self.ort_sess = ort.InferenceSession(onnx_path, providers=self._get_providers())
        # find input shape
        inp = self.ort_sess.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape  # e.g. (1,3,112,112)
        outp = self.ort_sess.get_outputs()[0]
        self.output_name = outp.name
        global EMBEDDING_DIM
        EMBEDDING_DIM = outp.shape[1] if len(outp.shape) > 1 else outp.shape[0]
        print(f"[+] Loaded ONNX model: {onnx_path}")
        print(f"    Input: {self.input_shape}, Output dim: {EMBEDDING_DIM}")

    def _get_providers(self):
        # prefer CPUExecutionProvider; if GPU available, ORT will pick suitable provider
        provs = ort.get_available_providers()
        # Use them as-is
        return provs

    def preprocess(self, face_bgr):
        # Model expects (1,H,W,C) = (1,112,112,3)
        h, w, c = self.input_shape[1], self.input_shape[2], self.input_shape[3]  # 112,112,3
        
        # Enhanced lighting normalization using CLAHE (better for varied lighting)
        # Convert to LAB color space for better lighting handling
        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        
        # Merge back and convert to BGR
        lab_clahe = cv2.merge([l_clahe, a, b])
        face_bgr_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        # Use high-quality interpolation for better resizing
        face = cv2.resize(face_bgr_enhanced, (w, h), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        tensor = rgb[None, :, :, :].astype(np.float32)  # (1, H, W, C)
        return tensor



    def embed(self, face_bgr):
        inp = self.preprocess(face_bgr)
        out = self.ort_sess.run([self.output_name], {self.input_name: inp})[0][0]
        # L2-normalize
        vec = out.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    
    def is_good_quality_face(self, face_bgr, min_size=80):
        """Check if the face is of sufficient quality for recognition with enhanced checks"""
        if face_bgr is None or face_bgr.size == 0:
            return False
        
        h, w = face_bgr.shape[:2]
        
        # Check minimum size - increased threshold
        if h < min_size or w < min_size:
            return False
        
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        
        # Enhanced blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Stricter blur threshold
            return False
        
        # Additional blur check using gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx**2 + gy**2).mean()
        if gradient_magnitude < 15:  # Low gradient means blurry
            return False
        
        # Check brightness with acceptable range
        mean_brightness = np.mean(gray)
        if mean_brightness < 30 or mean_brightness > 225:
            return False
        
        # Check contrast - reject low contrast images
        contrast = gray.std()
        if contrast < 20:  # Low contrast
            return False
        
        # Edge density check - good faces should have clear edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density < 0.05 or edge_density > 0.4:  # Too few or too many edges
            return False
            
        return True

# -----------------------------
# ANN index (Annoy)
# -----------------------------
class VectorIndex:
    def __init__(self, dim, path=ANNOY_INDEX_PATH):
        self.dim = dim
        self.path = path
        self.index = AnnoyIndex(dim, 'angular')  # angular ~ cosine
        self.id_map = []  # DB id -> index position (order)
        self.loaded = False

    def build_from_db(self, ids, embeddings):
        self.index = AnnoyIndex(self.dim, 'angular')
        self.id_map = []
        for i, emb in enumerate(embeddings):
            self.index.add_item(i, emb.tolist())
            self.id_map.append(ids[i])
        if len(embeddings) > 0:
            self.index.build(ANNOY_TREE_COUNT)
            self.index.save(self.path)
        self.loaded = True

    def load(self):
        if os.path.exists(self.path):
            self.index = AnnoyIndex(self.dim, 'angular')
            self.index.load(self.path)
            # id_map must be reconstructed from DB (caller will set it)
            self.loaded = True
        else:
            self.loaded = False

    def query(self, emb, top_k=5):
        if not self.loaded:
            return []
        idxs, dists = self.index.get_nns_by_vector(emb.tolist(), top_k, include_distances=True)
        # distances are angular distances (0-2). To map to cosine: cos_sim = 1 - dist^2/2 approx; we'll compute exact cosine separately
        return idxs, dists

# -----------------------------
# CLI operations
# -----------------------------
def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    create_db(conn)
    return conn

def rebuild_index(conn, vec_index):
    ids, names, embeddings = load_all_embeddings(conn)
    if len(embeddings) == 0:
        # empty index
        vec_index.id_map = []
        vec_index.loaded = False
        if os.path.exists(ANNOY_INDEX_PATH):
            os.remove(ANNOY_INDEX_PATH)
        return
    vec_index.build_from_db(ids, embeddings)
    # id_map already set to ids inside build_from_db
    print(f"[+] Rebuilt Annoy index with {len(embeddings)} items.")

def add_face_auto_flow(args, embedder, detector):
    """Automatic face capture with guided instructions"""
    conn = ensure_db()
    name = args.name if args.name else input("Name for the face: ").strip()
    cam = cv2.VideoCapture(args.camera if args.camera is not None else 0)
    
    # Instructions for different poses
    instructions = [
        {"text": "Look straight at the camera", "duration": 5, "buffer": 2},
        {"text": "Please remove glasses/specs if wearing any", "duration": 8, "buffer": 3},
        {"text": "Turn your face slightly LEFT", "duration": 4, "buffer": 2},
        {"text": "Turn your face slightly RIGHT", "duration": 4, "buffer": 2},
        {"text": "Tilt your head slightly UP", "duration": 4, "buffer": 2},
        {"text": "Tilt your head slightly DOWN", "duration": 4, "buffer": 2},
        {"text": "Look straight again - final capture", "duration": 4, "buffer": 2}
    ]
    
    samples = []
    current_instruction = 0
    start_time = time.time()
    buffer_phase = True  # Start with buffer time for each instruction
    
    print("[*] Automatic face capture starting. Follow the on-screen instructions.")
    print("[*] Press 'q' to quit, 's' to skip current instruction")
    
    while current_instruction < len(instructions) and cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("[!] Failed to grab frame")
            break
            
        display = frame.copy()
        current_time = time.time()
        elapsed = current_time - start_time
        
        instruction = instructions[current_instruction]
        
        # Determine if we're in buffer phase or capture phase
        if buffer_phase:
            remaining_time = instruction["buffer"] - elapsed
            if remaining_time <= 0:
                buffer_phase = False
                start_time = current_time  # Reset timer for capture phase
                elapsed = 0
            else:
                # Show preparation message
                cv2.putText(display, f"GET READY: {instruction['text']}", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(display, f"Preparing... {remaining_time:.1f}s", 
                           (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # Capture phase
            remaining_time = instruction["duration"] - elapsed
            
            if remaining_time <= 0:
                # Move to next instruction
                current_instruction += 1
                buffer_phase = True
                start_time = current_time
            else:
                # Show current instruction
                cv2.putText(display, instruction["text"], 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                cv2.putText(display, f"Hold position: {remaining_time:.1f}s", 
                           (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Try to capture face every 0.5 seconds during instruction
                if int(elapsed * 2) != int((elapsed - 0.033) * 2):  # Roughly every 0.5s
                    boxes = detector.detect(frame)
                    if len(boxes) > 0:
                        # Pick largest face
                        boxes_sorted = sorted(boxes, key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
                        x1, y1, x2, y2 = boxes_sorted[0]
                        face = frame[y1:y2, x1:x2]
                        
                        # Check face quality
                        if embedder.is_good_quality_face(face):
                            samples.append(face)
                            cv2.putText(display, f"✓ Captured! ({len(samples)} samples)", 
                                       (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw face detection boxes
        boxes = detector.detect(frame)
        for b in boxes:
            x1, y1, x2, y2 = b
            face = frame[y1:y2, x1:x2]
            if embedder.is_good_quality_face(face):
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for good quality
            else:
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for poor quality
        
        # Show progress
        cv2.putText(display, f"Step {current_instruction + 1}/{len(instructions)}", 
                   (20, display.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Samples collected: {len(samples)}", 
                   (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Automatic Face Capture - press 'q' to quit, 's' to skip", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Skip current instruction
            current_instruction += 1
            buffer_phase = True
            start_time = current_time
    
    cam.release()
    cv2.destroyAllWindows()
    
    if len(samples) == 0:
        print("[!] No samples captured, aborting.")
        conn.close()
        return
    
    print(f"[+] Captured {len(samples)} face samples automatically")
    
    # Compute embeddings and average them
    embs = []
    for s in samples:
        embs.append(embedder.embed(s))
    embedding = np.mean(np.stack(embs, axis=0), axis=0)
    embedding = embedding / np.linalg.norm(embedding)
    
    # Save to DB
    fid = add_face_to_db(conn, name, embedding, metadata={"samples": len(samples), "auto_capture": True})
    print(f"[+] Added face '{name}' with DB id {fid} using automatic capture")
    
    # Rebuild index
    vec_index = VectorIndex(dim=embedding.shape[0])
    rebuild_index(conn, vec_index)
    conn.close()

def add_face_flow(args, embedder, detector):
    conn = ensure_db()
    name = args.name if args.name else input("Name for the face: ").strip()
    cam = cv2.VideoCapture(args.camera if args.camera is not None else 0)
    print("[*] Press SPACE to capture a face sample. Capture multiple angles for better robustness. Press 'q' to finish.")
    samples = []
    while True:
        ret, frame = cam.read()
        if not ret:
            print("[!] Failed to grab frame")
            break
        display = frame.copy()
        boxes = detector.detect(frame)
        for b in boxes:
            x1,y1,x2,y2 = b
            cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.imshow("Add Face - press SPACE to capture, q to quit", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # pick largest face if multiple
            if len(boxes) == 0:
                print("[!] No face detected in capture. Try again.")
                continue
            boxes_sorted = sorted(boxes, key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
            x1,y1,x2,y2 = boxes_sorted[0]
            face = frame[y1:y2, x1:x2]
            
            # Check face quality before adding
            if not embedder.is_good_quality_face(face):
                print("[!] Face quality too low (blurry, too dark/bright, or too small). Try again with better lighting and focus.")
                continue
                
            samples.append(face)
            print(f"[+] Captured high-quality sample #{len(samples)}")
        elif key == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    if len(samples) == 0:
        print("[!] No samples captured, aborting.")
        return
    # compute embeddings and average them
    embs = []
    for s in samples:
        embs.append(embedder.embed(s))
    embedding = np.mean(np.stack(embs, axis=0), axis=0)
    embedding = embedding / np.linalg.norm(embedding)
    # save to DB
    fid = add_face_to_db(conn, name, embedding, metadata={"samples": len(samples)})
    print(f"[+] Added face '{name}' with DB id {fid}")
    # rebuild index
    vec_index = VectorIndex(dim=embedding.shape[0])
    rebuild_index(conn, vec_index)
    conn.close()

def list_faces(args):
    conn = ensure_db()
    c = conn.cursor()
    c.execute("SELECT id, name, metadata, created_at FROM faces ORDER BY id")
    rows = c.fetchall()
    if not rows:
        print("[*] No faces in database.")
    else:
        print(f"\n[*] Found {len(rows)} face(s) in database:")
        for r in rows:
            metadata = json.loads(r[2]) if r[2] else {}
            samples_info = f" ({metadata.get('samples', 'unknown')} samples)" if metadata else ""
            auto_info = " [AUTO]" if metadata.get('auto_capture') else " [MANUAL]"
            print(f"  ID: {r[0]} | Name: {r[1]}{samples_info}{auto_info} | Added: {r[3]}")
    conn.close()

def delete_face(args):
    """Delete a face from the database"""
    conn = ensure_db()
    c = conn.cursor()
    
    # First, show all faces
    c.execute("SELECT id, name, metadata, created_at FROM faces ORDER BY id")
    rows = c.fetchall()
    if not rows:
        print("[*] No faces in database to delete.")
        conn.close()
        return
    
    print("\n[*] Current faces in database:")
    for r in rows:
        metadata = json.loads(r[2]) if r[2] else {}
        samples_info = f" ({metadata.get('samples', 'unknown')} samples)" if metadata else ""
        auto_info = " [AUTO]" if metadata.get('auto_capture') else " [MANUAL]"
        print(f"  ID: {r[0]} | Name: {r[1]}{samples_info}{auto_info} | Added: {r[3]}")
    
    # Get user input for deletion
    if args.id:
        face_id = args.id
    elif args.name:
        # Find by name
        c.execute("SELECT id FROM faces WHERE name = ? COLLATE NOCASE", (args.name,))
        result = c.fetchone()
        if not result:
            print(f"[!] No face found with name '{args.name}'")
            conn.close()
            return
        face_id = result[0]
    else:
        try:
            user_input = input("\nEnter ID or Name to delete (or 'cancel' to abort): ").strip()
            if user_input.lower() == 'cancel':
                print("[*] Deletion cancelled.")
                conn.close()
                return
            
            # Try to parse as ID first
            try:
                face_id = int(user_input)
            except ValueError:
                # Treat as name
                c.execute("SELECT id FROM faces WHERE name = ? COLLATE NOCASE", (user_input,))
                result = c.fetchone()
                if not result:
                    print(f"[!] No face found with name '{user_input}'")
                    conn.close()
                    return
                face_id = result[0]
        except KeyboardInterrupt:
            print("\n[*] Deletion cancelled.")
            conn.close()
            return
    
    # Verify the face exists and get details
    c.execute("SELECT id, name FROM faces WHERE id = ?", (face_id,))
    face_to_delete = c.fetchone()
    if not face_to_delete:
        print(f"[!] No face found with ID {face_id}")
        conn.close()
        return
    
    # Confirm deletion
    face_name = face_to_delete[1]
    if not args.force:
        try:
            confirm = input(f"Are you sure you want to delete '{face_name}' (ID: {face_id})? [y/N]: ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("[*] Deletion cancelled.")
                conn.close()
                return
        except KeyboardInterrupt:
            print("\n[*] Deletion cancelled.")
            conn.close()
            return
    
    # Delete the face
    c.execute("DELETE FROM faces WHERE id = ?", (face_id,))
    conn.commit()
    
    print(f"[+] Deleted face '{face_name}' (ID: {face_id})")
    
    # Rebuild the index since we removed a face
    vec_index = VectorIndex(dim=EMBEDDING_DIM if EMBEDDING_DIM else 512)  # Use default dim if not set
    rebuild_index(conn, vec_index)
    
    conn.close()

def export_db(args):
    # dump embeddings and names to JSON+npz
    conn = ensure_db()
    ids, names, embeddings = load_all_embeddings(conn)
    if len(embeddings) == 0:
        print("[!] No faces to export.")
        return
    np.savez_compressed("faces_embeddings.npz", ids=np.array(ids), names=np.array(names), embeddings=np.stack(embeddings))
    print("[+] Exported to faces_embeddings.npz")
    conn.close()

def recognize_flow(args, embedder, detector):
    # load DB & index
    conn = ensure_db()
    ids, names, embeddings = load_all_embeddings(conn)
    if len(embeddings) == 0:
        print("[!] No faces in database. Add faces first.")
        conn.close()
        return
    dim = embeddings[0].shape[0]
    vec_index = VectorIndex(dim=dim)
    # build index state from DB
    vec_index.build_from_db(ids, embeddings)
    # map index position -> (db id, name)
    position_to_meta = {i: (ids[i], names[i]) for i in range(len(ids))}
    cam = cv2.VideoCapture(args.camera if args.camera is not None else 0)
    
    # Variables for handling the 5-second pause after recognition
    last_recognition_time = 0
    pause_duration = 5.0  # seconds
    unlocked_person = None
    
    # Temporal smoothing - track recognition history
    recognition_history = []  # List of (name, confidence) tuples
    
    print(f"[*] Starting recognition with {int(SIMILARITY_THRESHOLD*100)}% threshold. Press 'q' to quit.")
    print("[*] System will pause for 5 seconds after recognizing a person.")
    print(f"[*] Enhanced detection: min face size={MIN_FACE_SIZE}px, temporal frames={TEMPORAL_FRAMES}")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
            
        current_time = time.time()
        
        # Check if we're in the pause period after a successful recognition
        if current_time - last_recognition_time < pause_duration and unlocked_person:
            # Display the unlocked message during pause
            cv2.putText(frame, f"UNLOCKED - {unlocked_person}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Paused for {pause_duration - (current_time - last_recognition_time):.1f}s", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Face Recognition System - press q to exit", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            continue
        else:
            # Reset after pause period
            unlocked_person = None
            
        boxes = detector.detect(frame)
        recognition_made = False
        best_match_this_frame = None
        best_match_confidence = 0
        
        for b in boxes:
            x1,y1,x2,y2 = b
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
                
            # Check face quality before processing
            if not embedder.is_good_quality_face(face):
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)  # Red box for poor quality
                cv2.putText(frame, "Poor Quality", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                continue
                
            emb = embedder.embed(face)
            # query annoy
            idxs, dists = vec_index.query(emb, top_k=5)
            best = None
            best_sim = -1.0
            best_name = None
            # compute cosine similarity with the candidates exactly
            for rank_i, idx in enumerate(idxs):
                db_id, cand_name = position_to_meta.get(idx, (None, None))
                cand_emb = embeddings[idx]
                cos = float(np.dot(emb, cand_emb) / (np.linalg.norm(emb)*np.linalg.norm(cand_emb) + 1e-9))
                if cos > best_sim:
                    best_sim = cos
                    best = db_id
                    best_name = cand_name
            
            # Show similarity percentage even when not recognized (for debugging)
            if best_name and best_sim > 0:
                print(f"[DEBUG] Best match: {best_name} with {best_sim*100:.1f}% confidence")
            
            # Track best match for temporal smoothing
            if best_name and best_sim >= SIMILARITY_THRESHOLD:
                if best_sim > best_match_confidence:
                    best_match_this_frame = best_name
                    best_match_confidence = best_sim
                    
            # Check if similarity meets the threshold
            if best is not None and best_sim >= SIMILARITY_THRESHOLD:
                label = f"{best_name} ({best_sim*100:.1f}%)"
                box_color = (0, 255, 0)  # Green for recognized
                text_color = (0, 255, 0)
            else:
                label = f"Unknown ({best_sim*100:.1f}%)" if best_sim > 0 else "Unknown"
                box_color = (0, 0, 255)  # Red for unknown
                text_color = (0, 0, 255)
                
            # annotate
            cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Temporal smoothing: add current frame result to history
        if best_match_this_frame:
            recognition_history.append((best_match_this_frame, best_match_confidence))
        else:
            recognition_history.append((None, 0))
        
        # Keep only last TEMPORAL_FRAMES frames
        if len(recognition_history) > TEMPORAL_FRAMES:
            recognition_history.pop(0)
        
        # Check if we have consistent recognition over multiple frames
        if len(recognition_history) >= TEMPORAL_FRAMES:
            # Count occurrences of each person
            person_counts = {}
            for person, conf in recognition_history:
                if person:
                    person_counts[person] = person_counts.get(person, 0) + 1
            
            # Check if any person appears in majority of recent frames
            for person, count in person_counts.items():
                if count >= (TEMPORAL_FRAMES * 0.6):  # 60% of frames
                    # Trigger unlock only if not already unlocked
                    if not recognition_made and unlocked_person != person:
                        last_recognition_time = current_time
                        unlocked_person = person
                        recognition_made = True
                        avg_confidence = np.mean([c for p, c in recognition_history if p == person])
                        print(f"[+] UNLOCKED: {person} recognized with {avg_confidence*100:.1f}% avg confidence (verified {count}/{TEMPORAL_FRAMES} frames)")
                        recognition_history = []  # Reset history after unlock
                        break
            
        # Display system status
        status_text = f"Threshold: {SIMILARITY_THRESHOLD*100:.0f}% | Faces in DB: {len(embeddings)}"
        cv2.putText(frame, status_text, (10, frame.shape[0]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        cv2.imshow("Face Recognition System - press q to exit", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
            
    cam.release()
    cv2.destroyAllWindows()
    conn.close()

# -----------------------------
# Main & CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Lightweight face recognition system (add/recognize/list/export).")
    parser.add_argument("--model", type=str, default=DEFAULT_ONNX_PATH, help="Path to ONNX embedding model (required).")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default 0).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Add a new face from camera. Optionally pass --name.")
    p_add.add_argument("--name", type=str, help="Name for the face.")
    p_add_auto = sub.add_parser("addauto", help="Add a new face using automatic guided capture.")
    p_add_auto.add_argument("--name", type=str, help="Name for the face.")
    p_recog = sub.add_parser("recognize", help="Run live recognition from camera.")
    p_list = sub.add_parser("list", help="List saved faces in DB.")
    p_delete = sub.add_parser("delete", help="Delete a face from the database.")
    p_delete.add_argument("--id", type=int, help="ID of the face to delete.")
    p_delete.add_argument("--name", type=str, help="Name of the face to delete.")
    p_delete.add_argument("--force", action="store_true", help="Skip confirmation prompt.")
    p_export = sub.add_parser("export", help="Export embeddings to faces_embeddings.npz")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[!] ONNX model not found at {args.model}. See script comments for download instructions.")
        sys.exit(1)

    # load embedder & detector with higher confidence threshold for better accuracy
    embedder = ONNXEmbedder(args.model)
    detector = FaceDetector(conf_threshold=0.75)

    if args.cmd == "add":
        add_face_flow(args, embedder, detector)
    elif args.cmd == "addauto":
        add_face_auto_flow(args, embedder, detector)
    elif args.cmd == "recognize":
        recognize_flow(args, embedder, detector)
    elif args.cmd == "list":
        list_faces(args)
    elif args.cmd == "delete":
        delete_face(args)
    elif args.cmd == "export":
        export_db(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
