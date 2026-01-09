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

            # Get the very latest entry (the one currently being updated)
            sql = """
                SELECT Date, Time, direction_left, direction_right, 
                       CPU_temp, CPU_usage 
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
                    "CPU_usage": row[5]
                }
            return None
            
    except Exception as e:
        print(f"Database error: {e}")
        return None

async def event_stream():
    """
    SSE generator that watches for CHANGES in the active minute
    """
    print("SSE Stream (Aggregated) starting...")
    yield f"data: {json.dumps({'type': 'connected', 'message': 'Stream established'})}\n\n"
    
    # keep track of the last data we successfully sent
    last_sent_data = None
    
    try:
        while True:
            current_data = await asyncio.to_thread(get_latest_minute_data)
            
            if current_data and current_data != last_sent_data:
                
                print(f"[STREAM] Detected change at {current_data['Time']}. Sending update...")
                
                last_sent_data = current_data
                
                json_payload = json.dumps({
                    'type': 'data',
                    'count': 1,
                    'data': [current_data],
                    'timestamp': datetime.now().isoformat()
                })
                
                yield f"data: {json_payload}\n\n"
            
            # Heartbeat (keep connection alive)
            await asyncio.sleep(2)
            
    except asyncio.CancelledError:
        print("SSE Stream cancelled")
    except Exception as e:
        print(f"SSE Stream error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

@app.get("/api/stream")
async def stream_data():
    """
    SSE endpoint for 24/7 live data streaming
    No authentication required for read-only stream
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
    
    Date = node_input.data.split(',')[0]
    Time = node_input.data.split(',')[1]
    date_time_remember = Time[:5]
    Direction = node_input.data.split(',')[2]
    X_start = node_input.data.split(',')[3]
    X_end = node_input.data.split(',')[4]
    CPU_percent = node_input.data.split(',')[5]
    RAM_used_MB = node_input.data.split(',')[6]
    GPU_percent = node_input.data.split(',')[8]
    CPU_temp_C = node_input.data.split(',')[9]
    GPU_temp_C = node_input.data.split(',')[10]

    conn = sqlite3.connect('direction_evl.db')
    cursor = conn.cursor()
    sql_insert = "INSERT INTO directional_data VALUES (?,?,?,?,?,?,?,?,?)"
    cursor.execute(sql_insert, [Date, Time, Direction, X_start, X_end, CPU_percent, RAM_used_MB, CPU_temp_C, GPU_temp_C])
    conn.commit()
    conn.close()
    update_db(Date, date_time_remember)
    
    return {"status": "success", "message": "Data received and stored"}

@app.get("/api/get_viz_data")
def get_viz_data(request: Request):
    return update_status, fetch_db()

@app.get("/api/data")
def get_data(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return update_status, fetch_db()

def update_db(target_date, target_minute):
    global update_status
    count = 0

    sql_query = """
    SELECT Date, Time, Direction, CPU_usage, mem_usage, CPU_temp, GPU_temp
    FROM directional_data
    WHERE Date = ? AND substr(Time, 1, 5) = ?
    """

    conn = sqlite3.connect('direction_evl.db')
    cursor = conn.cursor()
    cursor.execute(sql_query, (target_date, target_minute))

    rows = cursor.fetchall()
    conn.close()

    dir_left_total = dir_right_total = 0
    cpu_usage_total = mem_usage_total = cpu_temp_total = gpu_temp_total = 0.0
    
    if rows:
        for row in rows:
            count += 1
            direction = row[2]
            cpu_usage_total += row[3]
            mem_usage_total += row[4]
            cpu_temp_total += row[5]
            gpu_temp_total += row[6]

            if direction == "Right":
                dir_right_total += 1
            else:
                dir_left_total += 1
        
        conn = sqlite3.connect('direction_evl_min.db')
        cursor = conn.cursor()

        sql_insert = """
        UPDATE directional_data
        SET CPU_usage = ?, mem_usage = ?, CPU_temp = ?, GPU_temp = ?, direction_left = ?, direction_right = ?
        WHERE Date = ? AND Time = ?
        """
        
        cursor.execute(sql_insert, (
            round(cpu_usage_total / count, 2),
            round(mem_usage_total / count, 2),
            round(cpu_temp_total / count, 2),
            round(gpu_temp_total / count, 2),
            dir_left_total,
            dir_right_total,
            target_date,
            target_minute
        ))

        if cursor.rowcount == 0:
            sql_insert = """
            INSERT INTO directional_data 
            (Date, Time, CPU_usage, mem_usage, CPU_temp, GPU_temp, direction_left, direction_right)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(sql_insert, (
                target_date,
                target_minute,
                round(cpu_usage_total / count, 2),
                round(mem_usage_total / count, 2),
                round(cpu_temp_total / count, 2),
                round(gpu_temp_total / count, 2),
                dir_left_total,
                dir_right_total
            ))

        conn.commit()
        conn.close()
        update_status = 1

def fetch_db():
    conn = None 
    try:
        conn = sqlite3.connect('direction_evl_min.db')
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM directional_data")
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]
        return {"data": result, "status": "success"}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
    finally:
        if conn:
            conn.close()