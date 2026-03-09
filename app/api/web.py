#app/api/web.py
from typing import List
from fastapi import APIRouter, HTTPException, File, Form, UploadFile

from app.services.inference_service import inference_service
from app.utils.files import save_upload_temp, safe_remove

router = APIRouter(prefix="/api", tags=["web"])

@router.post("/detect")
async def api_detect(
    file: UploadFile = File(...),
    checkboxes: List[str] = Form([]),
    lesionFeatures: List[str] = Form([]),
    lesionLocations: List[str] = Form([]),
    symptom_text: str = Form(""),
):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    temp_path = None
    try:
        temp_path = save_upload_temp(file, folder="uploads")

        # print("checkboxes:", checkboxes)
        # print("lesionFeatures:", lesionFeatures)
        # print("lesionLocations:", lesionLocations)
        # print("symptom_text:", symptom_text)

        out = inference_service.predict(
            image_path=temp_path,
            checkboxes=checkboxes,
            lesion_features=lesionFeatures,
            lesion_locations=lesionLocations,
            symptom_text=symptom_text,
        )

        return {
            "success": True,
            "prediction": {"label": out.label, "confidence": out.confidence},
            "images": {"original_b64": out.original_b64, "gradcam_b64": out.gradcam_b64},
            "prompt_used": out.prompt_used,
        }

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Cannot process request")
    finally:
        if temp_path:
            safe_remove(temp_path)
