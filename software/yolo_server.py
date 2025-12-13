# yolo_server.py  (A6000 GPU server)

from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import uvicorn
import time
import traceback

app = FastAPI()

print("[A6000] Loading YOLO11x on GPU...")
try:
    model = YOLO("yolo11x.pt")  # auto-download if missing
    print("[A6000] YOLO11x ready.")
except Exception as e:
    print("[A6000] FAILED to load YOLO11x:", e)
    traceback.print_exc()
    raise

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    """
    Receive an image, run YOLO11x on GPU, return counts + timing.
    """
    try:
        t0 = time.time()

        bytes_data = await file.read()
        img = Image.open(BytesIO(bytes_data)).convert("RGB")

        # device=0 => A6000 GPU
        results = model(img, device=0, verbose=False)[0]
        elapsed = time.time() - t0

        num_objects = len(results.boxes)
        num_vehicles = 0
        num_pedestrians = 0

        for box in results.boxes:
            cls = int(box.cls.cpu().item())
            name = results.names[cls].lower()
            if "car" in name or "truck" in name or "bus" in name:
                num_vehicles += 1
            if "person" in name:
                num_pedestrians += 1

        return {
            "num_objects": num_objects,
            "num_vehicles": num_vehicles,
            "num_pedestrians": num_pedestrians,
            "yolo_time_sec": elapsed,
        }

    except Exception as e:
        print("[A6000] ERROR during inference:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Only listen on localhost; Windows will tunnel to this.
    uvicorn.run("yolo_server:app", host="IP", port="PORT", reload=False) #TODO change IP annd port 
