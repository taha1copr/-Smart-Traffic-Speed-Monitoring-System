import sqlite3
import numpy as np
import os
import time
import logging

# Configure local logger for DB utils
logger = logging.getLogger(__name__)

# Try to get FEATURE_DIM from reid_utils to ensure consistency
try:
    from reid_utils import FEATURE_DIM
except ImportError:
    FEATURE_DIM = 512 # Fallback for OSNet

DB_PATH = "car_fingerprints.db"

# ----------------------------------------------------
# Database Initialization
# ----------------------------------------------------
def ensure_db():
    """
    Ensures that the vehicle fingerprint database and tables exist.
    If the structure is old, it migrates it to the Multi-View schema.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check for modern table structure
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='car_features'")
    if not cursor.fetchone():
        logger.info("MIGRATION: Initializing new DB structure for Multi-View Re-ID...")
        
        # Cleanup old tables if any
        cursor.execute("DROP TABLE IF EXISTS cars")
        cursor.execute("DROP TABLE IF EXISTS car_features")

        # Parent Cars table: Stores unique vehicle identity
        cursor.execute("""
            CREATE TABLE cars (
                car_id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_seen_video TEXT,
                first_seen_frame INTEGER,
                last_seen_timestamp REAL
            )
        """)

        # Gallery table: Stores multiple feature vectors (views) for each car
        cursor.execute("""
            CREATE TABLE car_features (
                feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
                car_id INTEGER,
                embedding BLOB,
                created_at REAL,
                FOREIGN KEY(car_id) REFERENCES cars(car_id)
            )
        """)
        logger.info("DB: SQLite Database Initialized.")

    conn.commit()
    conn.close()

# ----------------------------------------------------
# Byte Encoding / Decoding for Embeddings
# ----------------------------------------------------
def encode_embedding(emb: np.ndarray) -> bytes:
    """Converts a numpy array to bytes for SQL BLOB storage."""
    emb = np.asarray(emb, dtype=np.float32)
    return emb.tobytes()

def decode_embedding(blob: bytes) -> np.ndarray:
    """Converts SQL BLOB back to a numpy float32 array."""
    if blob is None: return None
    return np.frombuffer(blob, dtype=np.float32)

# ----------------------------------------------------
# Vehicle Persistence
# ----------------------------------------------------
def save_car(video_path: str, frame_no: int, bbox: tuple, embedding: np.ndarray) -> int:
    """
    Registers a new vehicle in the system and saves its first embedding.
    Returns the newly created car_id.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Create Parent Identity
    cursor.execute("""
        INSERT INTO cars (first_seen_video, first_seen_frame, last_seen_timestamp)
        VALUES (?, ?, ?)
    """, (video_path, frame_no, time.time()))

    car_id = cursor.lastrowid

    # 2. Save First View (Embedding)
    cursor.execute("""
        INSERT INTO car_features (car_id, embedding, created_at)
        VALUES (?, ?, ?)
    """, (car_id, encode_embedding(embedding), time.time()))

    conn.commit()
    conn.close()
    return car_id

# ----------------------------------------------------
# Multi-View Search & Comparison
# ----------------------------------------------------
def compare_embedding(embedding: np.ndarray, threshold: float = 0.75):
    """
    Performs Re-ID matching using Multi-View logic:
    - Calculates Max Cosine Similarity against all stored views for each car.
    
    Returns: (matched_car_id, best_similarity)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT car_id, embedding FROM car_features")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None, 0.0

    queryvec = np.asarray(embedding, dtype=np.float32)
    car_best_sims = {}

    for car_id, blob in rows:
        db_emb = decode_embedding(blob)
        if db_emb is None: continue
            
        # Dimension Check: Auto-reset if AI model changed
        if db_emb.size != queryvec.size:
            logger.error("DB: Dimension Mismatch detected! Wiping database to adapt to new model...")
            os.remove(DB_PATH)
            ensure_db()
            return None, 0.0

        # Since query and DB vectors are L2-normalized, Dot Product = Cosine Similarity
        similarity = float(np.dot(queryvec, db_emb))

        if car_id not in car_best_sims or similarity > car_best_sims[car_id]:
            car_best_sims[car_id] = similarity

    matched_id = None
    max_sim = 0.0

    # Pick the best car that exceeds threshold
    for car_id, sim in car_best_sims.items():
        if sim >= threshold and sim > max_sim:
            max_sim = sim
            matched_id = car_id

    if matched_id:
        return matched_id, max_sim
    else:
        # Return None but provide highest sim found for debugging
        global_max = max(car_best_sims.values()) if car_best_sims else 0.0
        return None, global_max

# ----------------------------------------------------
# Smart Gallery Management
# ----------------------------------------------------
def update_car_embedding(car_id: int, new_emb: np.ndarray):
    """
    Updates the car's gallery with the new embedding:
    - If it's near-identical to an existing view, blend them (Temporal Smoothing).
    - If it's a new unique angle, add it to gallery (up to MAX_SIZE).
    - If gallery is full, replace the oldest/least relevant view.
    """
    MAX_GALLERY = 5
    MERGE_LIMIT = 0.98 # Above this, we blend instead of adding a new entry

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Update activity heartbeat
    cursor.execute("UPDATE cars SET last_seen_timestamp=? WHERE car_id=?", (time.time(), car_id))

    cursor.execute("SELECT feature_id, embedding FROM car_features WHERE car_id=?", (car_id,))
    rows = cursor.fetchall()

    new_emb = np.asarray(new_emb, dtype=np.float32)
    best_sim = 0.0
    best_fid = -1
    
    gallery_data = []
    for fid, blob in rows:
        emb = decode_embedding(blob)
        sim = float(np.dot(new_emb, emb))
        gallery_data.append({'fid': fid, 'sim': sim, 'emb': emb})
        if sim > best_sim:
            best_sim = sim
            best_fid = fid

    if best_sim > MERGE_LIMIT:
        # High Similarity: Blend features (Moving Average)
        old_emb = next(item['emb'] for item in gallery_data if item['fid'] == best_fid)
        blended = (old_emb * 0.8) + (new_emb * 0.2)
        blended /= np.linalg.norm(blended) # Re-normalize
        
        cursor.execute("UPDATE car_features SET embedding=?, created_at=? WHERE feature_id=?",
                       (encode_embedding(blended), time.time(), best_fid))
    
    elif len(gallery_data) < MAX_GALLERY:
        # New Angle & Space available: Add new entry
        cursor.execute("INSERT INTO car_features (car_id, embedding, created_at) VALUES (?, ?, ?)",
                       (car_id, encode_embedding(new_emb), time.time()))
    
    else:
        # Gallery Full: Replace oldest entry
        cursor.execute("SELECT feature_id FROM car_features WHERE car_id=? ORDER BY created_at ASC LIMIT 1", (car_id,))
        oldest_fid = cursor.fetchone()[0]
        cursor.execute("UPDATE car_features SET embedding=?, created_at=? WHERE feature_id=?",
                       (encode_embedding(new_emb), time.time(), oldest_fid))

    conn.commit()
    conn.close()