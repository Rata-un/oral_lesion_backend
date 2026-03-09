from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from app.models.inference import process_with_ai_model, build_prompt_from_form

@dataclass
class InferenceResult:
    label: str
    confidence: float
    original_b64: str
    gradcam_b64: str
    prompt_used: str

class InferenceService:
    def predict(self, image_path: str, checkboxes: List[str], lesion_features: List[str], lesion_locations: List[str], symptom_text: str) -> InferenceResult:
        prompt = build_prompt_from_form(
            checkboxes=checkboxes,
            lesion_features=lesion_features,
            lesion_locations=lesion_locations,
            symptom_text=symptom_text,
        )

        original_b64, gradcam_b64, label, conf_str = process_with_ai_model(
            image_path=image_path,
            prompt_text=prompt,
        )

        if original_b64 is None or gradcam_b64 is None or label == "Error":
            raise RuntimeError("Error during AI processing")

        try:
            confidence = float(conf_str)
        except Exception:
            confidence = 0.0

        return InferenceResult(
            label=label,
            confidence=confidence,
            original_b64=original_b64,
            gradcam_b64=gradcam_b64,
            prompt_used=prompt,
        )

inference_service = InferenceService()
