import sqlite3
import os
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from typing import Dict, Any
from pydantic import BaseModel

load_dotenv()
app = FastAPI()
update_status = 0

class StatusInput(BaseModel):
    status: int

class nodeInput(BaseModel):
    name: str
    data: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

API_KEY = os.getenv("API_SECRET_KEY")
API_KEY_NAME = os.getenv("API_HEADER_NAME")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_latest_minute_data():
    """Fetch ONLY the single most recent minute row from the aggregated DB"""
    try:
        with sqlite3.connect('direction_evl_min.db') as conn:
            cursor = conn.cursor()

            # Updated to include GPU_usage
            sql = """
                SELECT Date, Time, direction_left, direction_right, 
                       CPU_temp, CPU_usage, GPU_usage, GPU_temp, mem_usage
                FROM directional_data 
                ORDER BY Date DESC, Time DESC 
                LIMIT 1
            """
            
            cursor.execute(sql)
            row = cursor.fetchone()
            
            if row:
                return {
                    "Date": row[0],
                    "Time": row[1],
                    "direction_left": row[2],
                    "direction_right": row[3],
                    "CPU_temp": row[4],
                    "CPU_usage": row[5],
                    "GPU_usage": row[6],
                    "GPU_temp": row[7],
                    "mem_usage": row[8]
                }
            return None
            
    except Exception as e:
        print(f"Database error: {e}")
        return None

async def event_stream():
    print("SSE Stream (Aggregated) starting...")
    yield f"data: {json.dumps({'type': 'connected', 'message': 'Stream established'})}\n\n"
    
    last_sent_data = None
    
    try:
        while True:
            current_data = await asyncio.to_thread(get_latest_minute_data)
            
            if current_data:
                if current_data != last_sent_data:
                    print(f"[STREAM] Change detected at {current_data['Time']}. Sending...")
                    
                    last_sent_data = current_data
                    
                    json_payload = json.dumps({
                        'type': 'data',
                        'count': 1,
                        'data': [current_data],
                        'timestamp': datetime.now().isoformat()
                    })
                    yield f"data: {json_payload}\n\n"
            
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        print("SSE Stream cancelled")
    except Exception as e:
        print(f"SSE Stream error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

@app.get("/api/stream")
async def stream_data():
    """
    SSE endpoint for 24/7 live data streaming
    """
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no", 
        }
    )

@app.post("/api/send_data")
def send_data(node_input: nodeInput, api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    parts = node_input.data.split(',')

    Date = parts[0]
    Time = parts[1]
    date_time_remember = Time[:5]
    Direction = parts[2]
    X_start = parts[3]
    X_end = parts[4]
    
    def safe_float(val):
        try:
            return float(val) if val and str(val).strip() else 0.0
        except ValueError:
            return 0.0

    # Matching the 11-item array sent by your Jetson
    CPU_percent = safe_float(parts[5])
    GPU_percent = safe_float(parts[6])  # Now capturing GPU Usage
    RAM_used_MB = safe_float(parts[7])
    # parts[8] is RAM_total, which we ignore for the DB
    CPU_temp_C = safe_float(parts[9])
    GPU_temp_C = safe_float(parts[10]) 

    conn = sqlite3.connect('direction_evl.db')
    cursor = conn.cursor()
    
    # 10 Columns explicitly named to perfectly match direction_evl.db
    sql_insert = """
        INSERT INTO directional_data 
        (Date, Time, Direction, x_start, x_end, CPU_usage, GPU_usage, mem_usage, CPU_temp, GPU_temp) 
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """
    
    cursor.execute(sql_insert, [
        Date, Time, Direction, X_start, X_end, 
        CPU_percent, GPU_percent, RAM_used_MB, CPU_temp_C, GPU_temp_C
    ])
    
    conn.commit()
    conn.close()
    
    update_db(Date, date_time_remember)

    return {"status": "success", "message": "Data received and stored"}

@app.get("/api/get_viz_data")
def get_viz_data(request: Request):
    return update_status, fetch_db()

@app.post("/api/send_status")
def receive_status(status_input: StatusInput, api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    global update_status
    update_status = status_input.status
    
    print(f"Server received new status: {update_status}")
    
    return {"status": "success", "message": f"Program status updated to {update_status}"}

@app.get("/api/data")
def get_data(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return update_status, fetch_db()

def update_db(target_date, target_minute):
    """
    Rewritten to use SQLite's native math. 
    This automatically ignores -1 values and replaces the error-prone Python loop.
    """
    global update_status

    conn_raw = sqlite3.connect('direction_evl.db')
    cursor_raw = conn_raw.cursor()
    
    sql_query = """
    SELECT
        IFNULL(AVG(NULLIF(CPU_usage, -1)), -1),
        IFNULL(AVG(NULLIF(GPU_usage, -1)), -1),
        IFNULL(AVG(NULLIF(mem_usage, -1)), -1),
        IFNULL(AVG(NULLIF(CPU_temp, -1)), -1),
        IFNULL(AVG(NULLIF(GPU_temp, -1)), -1),
        SUM(CASE WHEN Direction = 'Right' THEN 1 ELSE 0 END),
        SUM(CASE WHEN Direction != 'Right' THEN 1 ELSE 0 END)
    FROM directional_data
    WHERE Date = ? AND substr(Time, 1, 5) = ?
    """
    
    cursor_raw.execute(sql_query, (target_date, target_minute))
    row = cursor_raw.fetchone()
    conn_raw.close()

    if row and row[0] is not None:
        # Extract rounded results
        avg_cpu = round(row[0], 2)
        avg_gpu = round(row[1], 2)
        avg_mem = round(row[2], 2)
        avg_cpu_temp = round(row[3], 2)
        avg_gpu_temp = round(row[4], 2)
        right_count = row[5] or 0
        left_count = row[6] or 0

        conn_min = sqlite3.connect('direction_evl_min.db')
        cursor_min = conn_min.cursor()

        sql_upsert = """
        INSERT OR REPLACE INTO directional_data
        (Date, Time, CPU_usage, GPU_usage, mem_usage, CPU_temp, GPU_temp, direction_left, direction_right)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor_min.execute(sql_upsert, (
            target_date, target_minute, 
            avg_cpu, avg_gpu, avg_mem, avg_cpu_temp, avg_gpu_temp, 
            left_count, right_count
        ))

        conn_min.commit()
        conn_min.close()
        
        update_status = 1

def fetch_db():
    conn = None 
    try:
        conn = sqlite3.connect('direction_evl_min.db')
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM directional_data ORDER BY Date DESC, Time DESC")
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]
        return {"data": result, "status": "success"}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
    finally:
        if conn:
            conn.close()
