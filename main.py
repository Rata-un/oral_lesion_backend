import os, sys, uuid, shutil
from typing import List
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from inference import process_with_ai_model, build_prompt_from_form

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

@app.get("/")
def health_check():
    return {"status": "i'm ok"}

@app.post("/api/detect")
async def api_detect(
    file: UploadFile = File(...),
    checkboxes: List[str] = Form([]),
    symptom_text: str = Form(""),
):
    # 1) เช็คว่าเป็นไฟล์รูปไหม
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    # 2) เซฟไฟล์ชั่วคราว
    temp_filepath = os.path.join("uploads", f"{uuid.uuid4()}_{file.filename}")
    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception:
        raise HTTPException(status_code=500, detail="Cannot save uploaded file")

    # 3) สร้าง prompt จาก checkbox + symptom_text (ใช้ฟังก์ชันจาก inference.py)
    final_prompt = build_prompt_from_form(checkboxes, symptom_text)

    # 4) เรียกโมเดล
    image_b64, gradcam_b64, name_out, eva_output = process_with_ai_model(
        image_path=temp_filepath,
        prompt_text=final_prompt,
    )

    # 5) ลบไฟล์ชั่วคราว
    try:
        os.remove(temp_filepath)
    except Exception:
        pass

    # 6) เช็ค error จากฝั่งโมเดล
    if image_b64 is None or gradcam_b64 is None or name_out == "Error":
        raise HTTPException(status_code=500, detail="Error during AI processing")

    # 7) แปลง confidence string -> float
    try:
        confidence_value = float(eva_output)
    except Exception:
        confidence_value = 0.0

    # 8) ส่ง JSON กลับไปให้ frontend
    return {
        "success": True,
        "prediction": {
            "label": name_out,
            "confidence": confidence_value,
        },
        "images": {
            "original_b64": image_b64,
            "gradcam_b64": gradcam_b64,
        },
        "prompt_used": final_prompt,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)