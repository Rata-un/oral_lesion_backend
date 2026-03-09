#app/utils/images.py
import base64
import os
from typing import Optional

def save_b64_jpg(b64_str: str, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if "," in b64_str and "base64" in b64_str[:50]:
        b64_str = b64_str.split(",", 1)[1]

    data = base64.b64decode(b64_str)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path
