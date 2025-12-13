from fastapi import FastAPI, UploadFile, File
import uvicorn
import requests
import time

# This is the tunneled A6000 YOLO server
DOWNSTREAM_URL = "http://IP:PORT/infer" #TODO change IP and PORT

app = FastAPI()

@app.get("/ping")
def ping():
    return {"msg": "pong from relay"}

@app.post("/infer")
async def relay(file: UploadFile = File(...)):
    """
    Receive image from Jetson, forward to A6000 YOLO server via SSH tunnel,
    return YOLO result back to Jetson.
    """
    bytes_data = await file.read()
    files = {
        "file": ("image.jpg", bytes_data, file.content_type or "image/jpeg")
    }

    t0 = time.time()
    r = requests.post(DOWNSTREAM_URL, files=files, timeout=60)
    elapsed = time.time() - t0

    r.raise_for_status()
    data = r.json()
    data["relay_total_time_sec"] = elapsed
    return data

if __name__ == "__main__":
    uvicorn.run("relay_server:app", host="0.0.0.0", port=6000, reload=False)
