import torch
import json
import base64
import os
import numpy as np

from PIL import Image, ImageOps
from io import BytesIO
import matplotlib.cm as cm
import torch.nn as nn

from app.models.densenet.fusion import (
    get_tokenizer,
    get_transforms,
    DenseNet121Classifier,
    TextClassifier
)

# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Config
# =========================

FUSION_MODE = os.getenv("FUSION_MODE", "manual6040")  # "learnable" or "manual6040"
ALPHA = float(os.getenv("FUSION_ALPHA", "0.6"))

print(f"[AI] Fusion mode: {FUSION_MODE} | alpha={ALPHA}")

# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(__file__)

LABEL_MAP_PATH = os.path.join(BASE_DIR, "densenet", "label_map_fusion_densenet.json")
IMG_WEIGHTS = os.path.join(BASE_DIR, "densenet", "best_densenet121_img.pth")
TXT_WEIGHTS = os.path.join(BASE_DIR, "densenet", "best_text.pth")
FUSION_WEIGHTS = os.path.join(BASE_DIR, "densenet", "best_fusion_densenet.pth")

# =========================
# Load label map
# =========================

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

class_names = [label for label, _ in sorted(label_map.items(), key=lambda x: x[1])]
NUM_CLASSES = len(class_names)

# =========================
# tokenizer + transform
# =========================

tokenizer = get_tokenizer()
transform = get_transforms((600, 600))

# =========================
# Model holders
# =========================

image_model = None
text_model = None
fusion_model = None

# =========================
# Load Models
# =========================

def load_models():

    global image_model, text_model, fusion_model

    if FUSION_MODE == "learnable":

        class FusionDenseNetText(nn.Module):

            def __init__(self, num_classes):
                super().__init__()

                self.image_model = DenseNet121Classifier(num_classes)
                self.text_model = TextClassifier(num_classes)

                self.fusion = nn.Sequential(
                    nn.Linear(num_classes * 2, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes)
                )

            def forward(self, image, ids, mask):

                img_logits = self.image_model(image)
                txt_logits = self.text_model(ids, mask)

                fused = torch.cat([img_logits, txt_logits], dim=1)

                return self.fusion(fused), img_logits, txt_logits

        fusion_model = FusionDenseNetText(NUM_CLASSES).to(device)
        fusion_model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=device))
        fusion_model.eval()

        image_model = fusion_model.image_model

    else:

        image_model = DenseNet121Classifier(NUM_CLASSES).to(device)
        text_model = TextClassifier(NUM_CLASSES).to(device)

        image_model.load_state_dict(torch.load(IMG_WEIGHTS, map_location=device))
        text_model.load_state_dict(torch.load(TXT_WEIGHTS, map_location=device))

        image_model.eval()
        text_model.eval()


# โหลดทันทีตอน import
load_models()

# =========================
# GradCAM++
# =========================

def find_last_conv2d(model):

    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m

    return None


def compute_gradcampp(img_pil, image_tensor, target_class):

    target_layer = find_last_conv2d(image_model)

    if target_layer is None:
        return None

    activations = []
    gradients = []

    def fwd_hook(m, i, o):
        activations.append(o)

    def bwd_hook(m, gi, go):
        gradients.append(go[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:

        image_model.zero_grad()

        logits = image_model(image_tensor)

        logits[0, target_class].backward(retain_graph=True)

        act = activations[-1][0]
        grad = gradients[-1][0]

        grad_sq = grad ** 2
        grad_cube = grad ** 3

        sum_act = act.sum(dim=(1, 2), keepdim=True)

        alpha = grad_sq / (2 * grad_sq + sum_act * grad_cube + 1e-8)

        weights = (alpha * torch.relu(grad)).sum(dim=(1, 2))

        cam = torch.relu(torch.sum(weights[:, None, None] * act, dim=0))

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        cam = cam.detach().cpu().numpy()

        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(img_pil.size)

        cam_np = np.asarray(cam_img) / 255.0

        heatmap = cm.get_cmap("jet")(cam_np)[:, :, :3]

        img_np = np.asarray(img_pil.convert("RGB")) / 255.0

        overlay = 0.6 * img_np + 0.4 * heatmap

        return (overlay * 255).astype(np.uint8)

    finally:

        h1.remove()
        h2.remove()

# =========================
# Inference Engines
# =========================

def run_learnable(image_tensor, ids, mask):

    with torch.inference_mode():

        fused_logits, _, _ = fusion_model(image_tensor, ids, mask)

        probs = torch.softmax(fused_logits, dim=1)[0]

    return probs.detach().cpu().numpy()


def run_manual(image_tensor, ids, mask):

    with torch.inference_mode():

        img_logits = image_model(image_tensor)
        txt_logits = text_model(ids, mask)

        probs_img = torch.softmax(img_logits, dim=1)
        probs_txt = torch.softmax(txt_logits, dim=1)

        fused = ALPHA * probs_img + (1 - ALPHA) * probs_txt

    return fused[0].detach().cpu().numpy()

# =========================
# Main API
# =========================

def process_with_ai_model(image_path, prompt_text):

    try:

        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)

        enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )

        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        if FUSION_MODE == "learnable":
            probs = run_learnable(image_tensor, ids, mask)
        else:
            probs = run_manual(image_tensor, ids, mask)

        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        gradcam_overlay = compute_gradcampp(image, image_tensor, pred_idx)

        def img_to_b64(img):

            buffer = BytesIO()
            img.save(buffer, format="JPEG")

            return base64.b64encode(buffer.getvalue()).decode()

        original_b64 = img_to_b64(image)

        if gradcam_overlay is not None:
            gradcam_b64 = img_to_b64(Image.fromarray(gradcam_overlay))
        else:
            gradcam_b64 = original_b64

        return original_b64, gradcam_b64, pred_label, f"{confidence:.2f}"

    except Exception as e:

        print("AI ERROR:", e)

        return None, None, "Error", "0.00"

# =========================
# Prompt Builder
# =========================

SYMPTOM_MAP = {
    "noSymptoms": "ไม่มีอาการ",
    "drinkAlcohol": "ดื่มเหล้า",
    "smoking": "สูบบุหรี่",
    "chewBetelNut": "เคี้ยวหมาก",
    "eatSpicyFood": "กินเผ็ดแสบ",
    "wipeOff": "เช็ดออกได้",
    "alwaysHurts": "เจ็บเมื่อโดนแผล",
}

LESION_FEATURE_MAP = {
    "whitePatch": "ปื้นสีขาว",
    "whiteReticular": "เส้นสีขาวคล้ายร่างแห",
    "whitePlaque": "ฝ้าขาว",
    "whiteRaisedLine": "เส้นสีขาวนูน",
    "whiteYellowLesion": "แผลสีขาวเหลือง",
}

LESION_LOCATION_MAP = {
    "tongue": "ลิ้น",
    "buccalMucosa": "กระพุ้งแก้ม",
    "palate": "เพดานปาก",
    "gingiva": "เหงือก",
    "lip": "ริมฝีปาก",
    "floorOfMouth": "พื้นปาก",
    "softTissue": "เยื่อบุริมฝีปากด้านใน",
}


def build_prompt_from_form(checkboxes, lesion_features, lesion_locations, symptom_text):

    parts = []

    checkboxes = checkboxes or []
    lesion_features = lesion_features or []
    lesion_locations = lesion_locations or []

    symptoms = {SYMPTOM_MAP.get(c) for c in checkboxes if SYMPTOM_MAP.get(c)}
    if symptoms:
        parts.append(" ".join(sorted(symptoms)))

    features = {LESION_FEATURE_MAP.get(f) for f in lesion_features if LESION_FEATURE_MAP.get(f)}
    if features:
        parts.append(" ".join(sorted(features)))

    locations = {LESION_LOCATION_MAP.get(l) for l in lesion_locations if LESION_LOCATION_MAP.get(l)}
    if locations:
        parts.append(" ".join(sorted(locations)))

    if symptom_text and symptom_text.strip():
        parts.append(symptom_text.strip())

    if not parts:
        return "ไม่มีอาการ"

    return "; ".join(parts)