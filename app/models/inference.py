import torch, random, json, base64, os
import numpy as np
from app.models.densenet.fusion import get_tokenizer, get_transforms, DenseNet121Classifier, TextClassifier
from PIL import Image, ImageOps
from io import BytesIO
import matplotlib.cm as cm
import torch.nn as nn
from typing import List

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#FUSION_LABELMAP_PATH = "densenet/label_map_fusion_densenet.json"
FUSION_LABELMAP_PATH = os.path.join(os.path.dirname(__file__), "densenet", "label_map_fusion_densenet.json")
#FUSION_WEIGHTS_PATH = "densenet/best_fusion_densenet.pth"
FUSION_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "densenet", "best_fusion_densenet.pth")

with open(FUSION_LABELMAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

# แปลง label_map -> list ชื่อคลาสเรียงตาม index
class_names = [label for label, _ in sorted(label_map.items(), key=lambda x: x[1])]
NUM_CLASSES = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
#   โมเดล FusionDenseNetText
# =========================

class FusionDenseNetText(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.image_model = DenseNet121Classifier(num_classes=num_classes)
        self.text_model = TextClassifier(num_classes=num_classes)
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, input_ids, attention_mask):
        logits_img = self.image_model(image)
        logits_txt = self.text_model(input_ids, attention_mask)
        fused_in = torch.cat([logits_img, logits_txt], dim=1)
        fused_out = self.fusion(fused_in)
        return fused_out, logits_img, logits_txt


fusion_model = FusionDenseNetText(num_classes=NUM_CLASSES).to(device)
fusion_model.load_state_dict(torch.load(FUSION_WEIGHTS_PATH, map_location=device))
fusion_model.eval()

tokenizer = get_tokenizer()
transform = get_transforms((600, 600))

# =========================
#   Grad-CAM
# =========================
def _find_last_conv2d(mod: torch.nn.Module):
    last = None
    for m in mod.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

def compute_gradcam_overlay(img_pil: Image.Image, image_tensor: torch.Tensor, target_class_idx: int):
    """
    สร้าง Grad-CAM overlay สำหรับภาพที่ input เข้ามา
    """
    img_branch = fusion_model.image_model
    target_layer = _find_last_conv2d(img_branch)
    if target_layer is None:
        return None

    activations = []
    gradients = []

    def fwd_hook(_m, _i, o):
        activations.append(o)

    def bwd_hook(_m, gin, gout):
        gradients.append(gout[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        img_branch.zero_grad()
        logits_img = img_branch(image_tensor)
        score = logits_img[0, target_class_idx]
        score.backward()

        act = activations[-1].detach()[0]  # [C, H, W]
        grad = gradients[-1].detach()[0]   # [C, H, W]

        # Global average pooling ที่ gradient -> weight ต่อ channel
        weights = torch.mean(grad, dim=(1, 2))
        cam = torch.relu(torch.sum(weights[:, None, None] * act, dim=0))

        # normalize [0,1]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        # resize cam ให้ขนาดเท่าภาพจริง
        cam_img = Image.fromarray((cam.cpu().numpy() * 255).astype(np.uint8)).resize(
            img_pil.size, Image.BILINEAR
        )
        cam_np = np.asarray(cam_img).astype(np.float32) / 255.0
        heatmap = cm.get_cmap("jet")(cam_np)[:, :, :3]

        img_np = np.asarray(img_pil.convert("RGB")).astype(np.float32) / 255.0
        overlay = (0.6 * img_np + 0.4 * heatmap)
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        return overlay
    finally:
        h1.remove()
        h2.remove()
        img_branch.zero_grad()

# =========================
#   ฟังก์ชันประมวลผลภาพ + ข้อความ
# =========================
def process_with_ai_model(image_path: str, prompt_text: str):
    """
    รับ path รูป + ข้อความอาการ -> ทำนายโรค + ส่งรูป original + grad-cam แบบ Base64
    """
    try:
        image_pil = Image.open(image_path)
        image_pil = ImageOps.exif_transpose(image_pil)
        image_pil = image_pil.convert("RGB")

        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            fused_logits, _, _ = fusion_model(image_tensor, ids, mask)
            probs_fused = torch.softmax(fused_logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs_fused))
        pred_label = class_names[pred_idx]
        confidence = float(probs_fused[pred_idx]) * 100.0

        gradcam_overlay_np = compute_gradcam_overlay(image_pil, image_tensor, pred_idx)

        def image_to_base64(img: Image.Image) -> str:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        original_b64 = image_to_base64(image_pil)

        if gradcam_overlay_np is not None:
            gradcam_pil = Image.fromarray(gradcam_overlay_np)
            gradcam_b64 = image_to_base64(gradcam_pil)
        else:
            gradcam_b64 = original_b64

        return original_b64, gradcam_b64, pred_label, f"{confidence:.2f}"
    except Exception as e:
        print(f"❌ Error during AI processing: {e}")
        return None, None, "Error", "0.00"


# =========================
#   สร้าง prompt จาก checkbox + ข้อความ
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

def build_prompt_from_form(
    checkboxes: List[str] | None,
    lesion_features: List[str] | None,
    lesion_locations: List[str] | None,
    symptom_text: str,
) -> str:
    """
    สร้าง prompt โดยใช้ logic เดียวกับฟังก์ชันที่คอมเมนต์ไว้ 
    คือใช้ Set operations เพื่อจัดการความขัดแย้งของอาการ และเชื่อมแต่ละกลุ่มด้วย semicolon
    """
    final_prompt_parts = []
    
    # ป้องกันกรณีค่าที่ส่งมาเป็น None
    checkboxes = checkboxes or []
    lesion_features = lesion_features or []
    lesion_locations = lesion_locations or []

    # ==========================================
    # 1. จัดการกลุ่มอาการและประวัติ (Symptoms & Habits)
    # ==========================================
    selected_symptoms_thai = {SYMPTOM_MAP.get(cb) for cb in checkboxes if SYMPTOM_MAP.get(cb)}

    if "ไม่มีอาการ" in selected_symptoms_thai:
        # ถ้าเลือก "ไม่มีอาการ" จะตัดอาการเจ็บ/แสบออก 
        # แต่ยังคงพวก พฤติกรรม (เหล้า/บุหรี่) และ การเช็ดออกได้ ไว้ตาม logic เดิม
        symptoms_group = {"เจ็บเมื่อโดนแผล", "กินเผ็ดแสบ"}
        
        # กรองเอาเฉพาะกลุ่มที่ไม่ใช่อาการเจ็บปวด
        final_selected_symptoms = selected_symptoms_thai - symptoms_group
        
        if final_selected_symptoms:
            final_prompt_parts.append(" ".join(sorted(list(final_selected_symptoms))))
            
    elif selected_symptoms_thai:
        final_prompt_parts.append(" ".join(sorted(list(selected_symptoms_thai))))

    # ==========================================
    # 2. จัดการลักษณะแผล (Lesion Features)
    # ==========================================
    selected_features_thai = {LESION_FEATURE_MAP.get(f) for f in lesion_features if LESION_FEATURE_MAP.get(f)}
    if selected_features_thai:
        final_prompt_parts.append(" ".join(sorted(list(selected_features_thai))))

    # ==========================================
    # 3. จัดการตำแหน่งแผล (Lesion Locations)
    # ==========================================
    selected_locations_thai = {LESION_LOCATION_MAP.get(l) for l in lesion_locations if LESION_LOCATION_MAP.get(l)}
    if selected_locations_thai:
        final_prompt_parts.append(" ".join(sorted(list(selected_locations_thai))))

    # ==========================================
    # 4. จัดการข้อความเพิ่มเติม (Free Text)
    # ==========================================
    if symptom_text and symptom_text.strip():
        final_prompt_parts.append(symptom_text.strip())

    # ==========================================
    # รวมผลลัพธ์ (Final Output)
    # ==========================================
    if not final_prompt_parts:
        return "ไม่มีอาการ"

    # เชื่อมแต่ละกลุ่มใหญ่ด้วย "; " เพื่อให้โมเดลแยกแยะส่วนประกอบได้ชัดเจน
    return "; ".join(final_prompt_parts)
    
