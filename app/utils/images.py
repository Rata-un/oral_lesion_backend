import base64
import os
from typing import Optional

def save_b64_jpg(b64_str: str, out_path: str) -> str:
    """
    รับ base64 (ไม่มี data:image/jpeg;base64, นำหน้า) แล้วเซฟเป็น jpg
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # กันกรณีบ64มี prefix ติดมา
    if "," in b64_str and "base64" in b64_str[:50]:
        b64_str = b64_str.split(",", 1)[1]

    data = base64.b64decode(b64_str)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path
