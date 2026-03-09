import os, json, hmac, hashlib, base64, uuid, httpx, asyncio
from dataclasses import dataclass, field
from typing import Dict, List

from fastapi import APIRouter, Request, Header, HTTPException
from fastapi.responses import FileResponse

from app.models.inference import build_prompt_from_form, process_with_ai_model
#from app.services.inference_service import InferenceService

from app.utils.images import save_b64_jpg

router = APIRouter(prefix="/line", tags=["line"])

UPLOAD_DIR = "line_uploads"
OUTPUT_DIR = "line_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@dataclass
class Session:
    state: str = "IDLE"
    image_path: str = ""
    symptoms: List[str] = field(default_factory=list)
    lesion_features: List[str] = field(default_factory=list)
    lesion_locations: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)

sessions: Dict[str, Session] = {}


#แบบที่ 1
# SYMPTOMS = [
#     ("noSymptoms", "ไม่มีอาการผิดปกติ"),
#     ("drinkAlcohol", "ดื่มเครื่องดื่มแอลกอฮอล์"),
#     ("smoking", "สูบบุหรี่"),
#     ("chewBetelNut", "เคี้ยวหมาก"),
#     ("eatSpicyFood", "กินเผ็ดแล้วระคายเคือง"),
#     ("wipeOff", "คราบขาวลอกออกได้"),
#     ("alwaysHurts", "เจ็บหรือระคายเคืองตลอดเวลา"),
# ]

#แบบที่ 2
SYMPTOMS = [
    (1, "noSymptoms", "ไม่มีอาการผิดปกติ"),
    (2, "drinkAlcohol", "ดื่มเครื่องดื่มแอลกอฮอล์"),
    (3, "smoking", "สูบบุหรี่"),
    (4, "chewBetelNut", "เคี้ยวหมาก"),
    (5, "eatSpicyFood", "กินเผ็ดแล้วระคายเคือง"),
    (6, "wipeOff", "คราบขาวที่เช็ดออกได้"),
    (7, "alwaysHurts", "เจ็บหรือระคายเคืองตลอดเวลา"),
]

LESION_FEATURES = [
    (1, "whitePatch", "ปื้นขาว"),
    (2, "whiteReticular", "เส้นสีขาวคล้ายร่างแห"),
    (3, "whitePlaque", "ฝ้าขาว"),
    (4, "whiteRaisedLine", "เส้นสีขาวนูน"),
    (5, "whiteYellowLesion", "แผลสีขาวเหลือง"),
]

LESION_LOCATIONS = [
    (1, "tongue", "ลิ้น"),
    (2, "buccalMucosa", "กระพุ้งแก้ม"),
    (3, "palate", "เพดานปาก"),
    (4, "gingiva", "เหงือก"),
    (5, "lip", "ริมฝีปาก"),
    (6, "floorOfMouth", "พื้นใต้ลิ้น"),
    (7, "softTissue", "เนื้อเยื่อ"),
]

NO_SYMPTOM = "noSymptoms"
BLOCK_WHEN_NO_SYMPTOM = {"eatSpicyFood", "alwaysHurts"}


def _verify_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    mac = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)

async def _reply(reply_token: str, messages: list, access_token: str) -> None:
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post(url, headers=headers, json={"replyToken": reply_token, "messages": messages})
        r.raise_for_status()

async def _get_image_content(message_id: str, access_token: str) -> bytes:
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.content

#แบบที่ 1 แบบย่อข้อความอาการ **แต่ตอนนี้ข้อความยังตกอยู่
# def _quick_reply_symptoms(selected: List[str]) -> dict:
#     items = []
#     for key, label in SYMPTOMS:
#         prefix = "✅ " if key in selected else ""
#         items.append({
#             "type": "action",
#             "action": {"type": "postback", "label": (prefix + label)[:20], "data": f"symptom:{key}", "displayText": label}
#         })
#     items.append({"type": "action", "action": {"type": "postback", "label": "ประมวลผล", "data": "do:run", "displayText": "process"}})
#     items.append({"type": "action", "action": {"type": "postback", "label": "เริ่มใหม่", "data": "do:reset", "displayText": "restart"}})
#     return {"items": items[:13]}

#แบบที่ 2 เพิ่มเลข---------

def _quick_reply_symptoms(selected: List[str]) -> dict:
    items = []
    for num, key, label in SYMPTOMS:
        prefix = "✅" if key in selected else ""
        items.append({
            "type": "action",
            "action": {
                "type": "postback",
                "label": f"{num}{prefix}"[:20],          # ปุ่มสั้นมาก
                "data": f"symptom:{key}",
                "displayText": f"{num}. {label}"         # โชว์ข้อความเต็มในแชตตอนกด
            }
        })

    items.append({
        "type": "action",
        "action": {"type": "postback", "label": "ประมวลผล", "data": "do:run", "displayText": "ประมวลผลเลย"}
    })
    items.append({
        "type": "action",
        "action": {"type": "postback", "label": "ถัดไป", "data": "next:feature", "displayText": "ถัดไปเลือกลักษณะแผล"}
    })
    items.append({
        "type": "action",
        "action": {"type": "postback", "label": "เริ่มใหม่", "data": "do:reset", "displayText": "เริ่มใหม่"}
    })
    return {"items": items[:13]}


def _symptom_list_text(selected: List[str]) -> str:
    lines = ["เลือกอาการ/ประวัติ โดยกดที่ปุ่มตัวเลขด้านล่าง:"]
    for num, key, label in SYMPTOMS:
        mark = "✅" if key in selected else "⬜"
        lines.append(f"{num}. {mark} {label}")
    lines.append("กด \"ประมวลผล\" เพื่อวิเคราะห์เลย หรือกด \"ถัดไป\" เพื่อระบุลักษณะและตำแหน่งแผลเพิ่มเติม")
    return "\n".join(lines)
#---------------------------

def _feature_list_text(selected: List[str]) -> str:
    lines = ["เลือกลักษณะแผล โดยกดที่ปุ่มตัวเลขด้านล่าง:"]
    for num, key, label in LESION_FEATURES:
        mark = "✅" if key in selected else "⬜"
        lines.append(f"{num}. {mark} {label}")
    lines.append("กด \"ประมวลผล\" เพื่อวิเคราะห์เลย หรือกด \"ถัดไป\" เพื่อระบุตำแหน่งแผลเพิ่มเติม")
    return "\n".join(lines)

def _location_list_text(selected: List[str]) -> str:
    lines = ["เลือกตำแหน่งแผล โดยกดที่ปุ่มตัวเลขด้านล่าง:"]
    for num, key, label in LESION_LOCATIONS:
        mark = "✅" if key in selected else "⬜"
        lines.append(f"{num}. {mark} {label}")
    lines.append("กดปุ่ม \"ประมวลผล\" เมื่อเลือกเสร็จ")
    return "\n".join(lines)

def _quick_reply_features(selected: List[str]) -> dict:
    items = []
    for num, key, label in LESION_FEATURES:
        prefix = "✅" if key in selected else ""
        items.append({
            "type": "action",
            "action": {
                "type": "postback",
                "label": f"{num}{prefix}",
                "data": f"feature:{key}",
                "displayText": f"{num}. {label}"
            }
        })

    items.append({
        "type": "action",
        "action": {"type": "postback", "label": "ประมวลผล", "data": "do:run", "displayText": "ประมวลผลเลย"}
    })
    items.append({
        "type": "action",
        "action": {"type": "postback", "label": "ถัดไป", "data": "next:location", "displayText": "ถัดไปเลือกตำแหน่งแผล"}
    })
    items.append({
        "type": "action",
        "action": {"type": "postback", "label": "เริ่มใหม่", "data": "do:reset", "displayText": "เริ่มใหม่"}
    })

    return {"items": items[:13]}

def _quick_reply_locations(selected: List[str]) -> dict:
    items = []

    for num, key, label in LESION_LOCATIONS:
        prefix = "✅" if key in selected else ""
        items.append({
            "type": "action",
            "action": {
                "type": "postback",
                "label": f"{num}{prefix}",
                "data": f"location:{key}",
                "displayText": f"{num}. {label}"
            }
        })

    items.append({
        "type": "action",
        "action": {"type": "postback", "label": "ประมวลผล", "data": "do:run", "displayText": "ประมวลผล"}
    })
    items.append({
        "type": "action",
        "action": {"type": "postback", "label": "เริ่มใหม่", "data": "do:reset", "displayText": "เริ่มใหม่"}
    })

    return {"items": items[:13]}


async def _delayed_cleanup(paths: list[str], delay_seconds: int = 300):
    await asyncio.sleep(delay_seconds)
    for p in paths:
        _safe_remove(p)

def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
            print(f"[LINE] deleted: {path}", flush=True)
    except Exception as e:
        print(f"[LINE] delete failed: {path} err={e}", flush=True)

def _cleanup_session_files(sess: Session) -> None:
    # ลบรูปที่อัปโหลด
    _safe_remove(sess.image_path)

    # ลบรูป output ที่เคยส่ง (เก็บไว้เป็น path เต็ม)
    for p in list(sess.output_files):
        _safe_remove(p)

    sess.image_path = ""
    sess.output_files = []

@router.get("/img/{filename}")
def get_img(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="image/jpeg")

@router.post("/webhook")
async def webhook(request: Request, x_line_signature: str = Header(None)):
    # ✅ อ่าน env “ตอนเรียกจริง” (ไม่พังจาก import order)
    channel_secret = os.getenv("LINE_CHANNEL_SECRET", "")
    access_token = os.getenv("LINE_ACCESS_TOKEN", "")
    public_base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

    if not channel_secret or not access_token:
        raise HTTPException(status_code=500, detail="Missing LINE envs")
    if not public_base_url:
        raise HTTPException(status_code=500, detail="Missing PUBLIC_BASE_URL")

    body = await request.body()
    if not x_line_signature or not _verify_signature(body, x_line_signature, channel_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = json.loads(body.decode("utf-8"))
    events = payload.get("events", [])

    for ev in events:
        reply_token = ev.get("replyToken")
        user_id = (ev.get("source") or {}).get("userId")
        if not reply_token or not user_id:
            continue

        sess = sessions.get(user_id) or Session()
        sessions[user_id] = sess

        ev_type = ev.get("type")

        # text
        if ev_type == "message" and (ev.get("message") or {}).get("type") == "text":
            text = ((ev["message"].get("text") or "").strip()).lower()

            if text == "start":
                _cleanup_session_files(sess)

                sess.state = "WAIT_IMAGE"
                sess.image_path = ""
                sess.symptoms = []
                await _reply(reply_token, [{"type": "text", "text": "เริ่มต้นการตรวจรอยโรค ✅\nส่งรูปรอยโรคในช่องปากมาได้เลยครับ"}], access_token)
                continue

            if text == "contactus":
                await _reply(reply_token, [{"type": "text", "text": "ติดต่อได้ที่\n65070195@kmitl.ac.th\n65070203@kmitl.ac.th"}], access_token)
                continue

            await _reply(reply_token, [{"type": "text", "text": "พิมพ์ start เพื่อเริ่มตรวจรอยโรค หรือ contactus เพื่อดูช่องทางติดต่อ"}], access_token)
            continue

        # image
        if ev_type == "message" and (ev.get("message") or {}).get("type") == "image":
            if sess.state != "WAIT_IMAGE":
                sess.state = "IDLE"
                await _reply(reply_token, [{"type": "text", "text": "พิมพ์ start ก่อนนะครับ แล้วค่อยส่งรูปอีกครั้ง"}], access_token)
                continue

            message_id = ev["message"]["id"]
            img_bytes = await _get_image_content(message_id, access_token)

            img_name = f"{uuid.uuid4()}.jpg"
            img_path = os.path.join(UPLOAD_DIR, img_name)
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            sess.image_path = img_path
            sess.state = "WAIT_SYMPTOMS"

            await _reply(reply_token, [{
                "type": "text",
#แบบ 1                "1 text": "ได้รับรูปแล้ว ✅\nกรุณาเลือกอาการ/ประวัติ แล้วกด “ประมวลผล”",
                "text": _symptom_list_text(sess.symptoms),
                "quickReply": _quick_reply_symptoms(sess.symptoms)
            }], access_token)
            continue

        # postback
        if ev_type == "postback":
            data = ((ev.get("postback") or {}).get("data") or "")

            if data.startswith("symptom:"):
                key = data.split(":", 1)[1]
                # ---- กดซ้ำ = ยกเลิก ----
                if key in sess.symptoms:
                    sess.symptoms.remove(key)
                    await _reply(reply_token, [{
                        "type": "text",
                        #1 "text": "อัปเดตอาการแล้ว ✅ กดเลือกต่อได้ แล้วกด “ประมวลผล”",
                        "text": _symptom_list_text(sess.symptoms),
                        "quickReply": _quick_reply_symptoms(sess.symptoms)
                    }], access_token)
                    continue

                # ---- กรณีจะ “เลือกเพิ่ม” ----

                # 1) ถ้าจะเลือก noSymptoms แต่มี eatSpicyFood/alwaysHurts อยู่ -> ห้าม
                if key == NO_SYMPTOM and any(s in sess.symptoms for s in BLOCK_WHEN_NO_SYMPTOM):
                    await _reply(reply_token, [{
                        "type": "text",
                        "text": "เลือก “ไม่มีอาการผิดปกติ” ไม่ได้ เพราะคุณเลือกอาการ “กินเผ็ดแล้วระคายเคือง” หรือ “เจ็บ/ระคายเคืองตลอดเวลา” อยู่\n(กดยกเลิกอาการนั้นก่อน)" + _symptom_list_text(sess.symptoms),
                        "quickReply": _quick_reply_symptoms(sess.symptoms)
                    }], access_token)
                    continue

                # 2) ถ้าเลือก eatSpicyFood หรือ alwaysHurts แต่มี noSymptoms อยู่ -> ห้าม
                if key in BLOCK_WHEN_NO_SYMPTOM and NO_SYMPTOM in sess.symptoms:
                    await _reply(reply_token, [{
                        "type": "text",
                        "text": "เลือกอาการนี้ไม่ได้ เพราะคุณเลือก “ไม่มีอาการผิดปกติ” อยู่\n(กดยกเลิก “ไม่มีอาการผิดปกติ” ก่อน)",
                        "quickReply": _quick_reply_symptoms(sess.symptoms)
                    }], access_token)
                    continue

                # 3) ผ่านเงื่อนไข -> เพิ่มเข้า list
                sess.symptoms.append(key)

                await _reply(reply_token, [{
                    "type": "text",
                    # 1 "text": "อัปเดตอาการแล้ว ✅ กดเลือกต่อได้ แล้วกด “ประมวลผล”",
                    "text": _symptom_list_text(sess.symptoms),
                    "quickReply": _quick_reply_symptoms(sess.symptoms)
                }], access_token)
                continue


            if data == "do:reset":
                _cleanup_session_files(sess)
                sess.state = "IDLE"
                sess.image_path = ""
                sess.symptoms = []
                sess.lesion_features = []
                sess.lesion_locations = []
                await _reply(reply_token, [{"type": "text", "text": "รีเซ็ตแล้วครับ ✅\nพิมพ์ start เพื่อเริ่มใหม่"}], access_token)
                continue

            if data.startswith("feature:"):
                key = data.split(":", 1)[1]
                if key in sess.lesion_features:
                    sess.lesion_features.remove(key)
                else:
                    sess.lesion_features.append(key)
                await _reply(reply_token, [{
                    "type": "text",
                    "text": _feature_list_text(sess.lesion_features),
                    "quickReply": _quick_reply_features(sess.lesion_features)
                }], access_token)
                continue

            if data == "next:feature":
                if not sess.image_path:
                    await _reply(reply_token, [{"type": "text", "text": "ยังไม่มีรูปครับ พิมพ์ start แล้วส่งรูปก่อนนะครับ"}], access_token)
                    continue
                sess.state = "WAIT_FEATURES"
                sess.lesion_features = []
                await _reply(reply_token, [{
                    "type": "text",
                    "text": _feature_list_text(sess.lesion_features),
                    "quickReply": _quick_reply_features(sess.lesion_features)
                }], access_token)
                continue

            if data == "next:location":
                sess.state = "WAIT_LOCATION"
                await _reply(reply_token, [{
                    "type": "text",
                    "text": _location_list_text(sess.lesion_locations),
                    "quickReply": _quick_reply_locations(sess.lesion_locations)
                }], access_token)
                continue

            if data.startswith("location:"):
                key = data.split(":", 1)[1]
                if key in sess.lesion_locations:
                    sess.lesion_locations.remove(key)
                else:
                    sess.lesion_locations.append(key)
                await _reply(reply_token, [{
                    "type": "text",
                    "text": _location_list_text(sess.lesion_locations),
                    "quickReply": _quick_reply_locations(sess.lesion_locations)
                }], access_token)
                continue

            if data == "do:run":
                if not sess.image_path:
                    await _reply(reply_token, [{"type": "text", "text": "ยังไม่มีรูปครับ พิมพ์ start แล้วส่งรูปก่อนนะครับ"}], access_token)
                    continue

                if sess.state not in ("WAIT_SYMPTOMS", "WAIT_FEATURES", "WAIT_LOCATION"):
                    await _reply(reply_token, [{"type": "text", "text": "ยังไม่มีรูปครับ พิมพ์ start แล้วส่งรูปก่อนนะครับ"}], access_token)
                    continue

                prompt = build_prompt_from_form(sess.symptoms, sess.lesion_features, sess.lesion_locations, "")
                print(f"[LINE] prompt_used={prompt!r} symptoms={sess.symptoms} features={sess.lesion_features} locations={sess.lesion_locations} image_path={sess.image_path}", flush=True)

                original_b64, gradcam_b64, label, conf_str = process_with_ai_model(sess.image_path, prompt)
                if original_b64 is None or gradcam_b64 is None or label == "Error":
                    await _reply(reply_token, [{"type": "text", "text": "ขออภัย ระบบประมวลผลผิดพลาด ลองใหม่อีกครั้งนะครับ"}], access_token)
                    continue

                _safe_remove(sess.image_path)
                sess.image_path = ""

                out_id = str(uuid.uuid4())
                original_file = f"{out_id}_original.jpg"
                gradcam_file = f"{out_id}_gradcam.jpg"
                save_b64_jpg(original_b64, os.path.join(OUTPUT_DIR, original_file))
                save_b64_jpg(gradcam_b64, os.path.join(OUTPUT_DIR, gradcam_file))

                sess.output_files = [
                    os.path.join(OUTPUT_DIR, original_file),
                    os.path.join(OUTPUT_DIR, gradcam_file),
                ]
                original_url = f"{public_base_url}/line/img/{original_file}"
                gradcam_url = f"{public_base_url}/line/img/{gradcam_file}"

                await _reply(reply_token, [
                    {"type": "image", "originalContentUrl": original_url, "previewImageUrl": original_url},
                    {"type": "image", "originalContentUrl": gradcam_url, "previewImageUrl": gradcam_url},
                    {"type": "text", "text": f"$ ผลการวิเคราะห์\nรอยโรคที่คาดการณ์: {label}\nโอกาสเป็นรอยโรค: {conf_str}%", "emojis": [{"index": 0, "productId": "5ac22b23040ab15980c9b44d", "emojiId": "031"}]},
                ], access_token)

                asyncio.create_task(_delayed_cleanup(list(sess.output_files), delay_seconds=300))

                sess.state = "IDLE"
                sess.image_path = ""
                sess.symptoms = []
                sess.lesion_features = []
                sess.lesion_locations = []
                continue

    return {"ok": True}
