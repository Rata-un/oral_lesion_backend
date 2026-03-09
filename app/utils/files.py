import os, uuid, shutil
from fastapi import UploadFile

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_upload_temp(file: UploadFile, folder: str = "uploads") -> str:
    ensure_dir(folder)
    filename = file.filename or "upload.bin"
    temp_path = os.path.join(folder, f"{uuid.uuid4()}_{filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path

def safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass
