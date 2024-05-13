from fastapi import FastAPI, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from static.scripts.python.pipeline import Pipeline
# import SimpleITK as sitk
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        print("load start")
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        print("load done")
    finally:
        upload_file.file.close()
        print("load finish")

async def create_upload_file(file: UploadFile):
    with open(os.path.join("upload_folder", file.filename), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}


@app.post("/upload")
async def upload_files(file1: UploadFile, file2: UploadFile):
    # Читаем содержимое файлов
    content1 = await file1.read()
    content2 = await file2.read()

    with open("data/ABUBAKAROVA.nii", "wb") as f1:
        f1.write(content1)
    
    with open("data/ABUBAKAROVA_label.nii", "wb") as f2:
        f2.write(content2)


    # Выводим содержимое файлов в консоль
    #print(f"File 1 content:\n{image}")
    #print(f"File 2 content:\n{mask}")



    if 1:
        return {"Result": 0}
    else:
        return {"Result": 0}