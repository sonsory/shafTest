import os
import time
from typing import Annotated
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse

from pathlib import Path
from app.models import mongodb
from app.models.book import BookModel
from app.book_scraper import NaverBookScraper


import numpy as np
from starlette.requests import Request

import io
import cv2
import pytesseract
import re
from pydantic import BaseModel
import shutil

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()


templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def read_img(img):
    # pytesseract.pytesseract.tesseract_cmd = "/app/.apt/usr/bin/tesseract"
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    text = pytesseract.image_to_string(img)
    return text


# 이미지를 저장할 디렉토리 경로
upload_dir = "uploaded_images"
os.makedirs(upload_dir, exist_ok=True)


app = FastAPI()


class ImageType(BaseModel):
    url: str


@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})


@app.post("/upf/")
async def upload_file(file: UploadFile = Form(...)):
    # 파일을 어딘가에 저장하거나 처리하는 작업을 수행할 수 있습니다.
    # 이 예제에서는 파일의 내용을 그대로 반환합니다.
    with open(f"{upload_dir}/{file.filename}", "wb") as image:
        shutil.copyfileobj(file.file, image)

    # 이미지 업로드 후 직접 리디렉션을 수행하여 업로드된 이미지 보여주기
    return RedirectResponse(url=f"/images/{file.filename}", status_code=303)
    # RedirectResponse의 status_code를 303 (See Other)로 설정하여 GET 요청으로 리디렉션을 수행
    # 이렇게 하면 이미지 업로드 후에 브라우저가 POST 요청을 다시 보내지 않고 GET 요청으로 이미지를 표시하게 됨.
    # 즉, 다시 설명하면, return RedirectResponse(url=f"/images/{file.filename}") 만 있다면,  POST 요청으로 리디렉션을 수행함으로써
    # INFO:     ::1:52665 - "POST /upf/ HTTP/1.1" 307 Temporary Redirect
    # INFO:     ::1:52665 - "POST /images/sdfsdfsdfdf.png HTTP/1.1" 405 Method Not Allowed
    # INFO:     ::1:52671 - "GET /images/sdfsdfsdfdf.png/image HTTP/1.1" 404 Not Found


@app.get("/images/{filename}", response_class=HTMLResponse)
async def show_image(request: Request, filename: str):
    image_path = f"{upload_dir}/{filename}"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return templates.TemplateResponse(
        "show_image.html", {"request": request, "image_path": filename}
    )


@app.get("/images/{filename}/image", response_class=FileResponse)
async def get_uploaded_image(filename: str):
    image_path = f"{upload_dir}/{filename}"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


# @app.get("/images/{filename}")
# async def get_uploaded_image(filename: str):
#     image_path = f"{upload_dir}/{filename}"
#     return FileResponse(image_path)


@app.post("/predict")
def prediction(request: Request, file: bytes = File(...)):
    if request.method == "POST":
        image_stream = io.BytesIO(file)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        label = read_img(frame)
        return label

    else:
        return "No post request found"


@app.get("/predic")
async def main(request: Request):
    file = """
            <body>
            <form action="/files/" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit">
            </form>
            </body>
            """
    if request.method == "POST":
        image_stream = io.BytesIO(file)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        label = read_img(frame)

        return templates.TemplateResponse(
            "index.html", {"request": request, "title": label}
        )
    else:
        return "No post request found"


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    # book = BookModel(keyword="파이썬", publisher="BJ퍼플릭", price=1200, image="me.png")
    # print(await mongodb.engine.save(book))
    return templates.TemplateResponse(
        "index.html", {"request": request, "title": "우리동네 도서관 feat.샤프코드"}
    )


@app.get("/dong", response_class=HTMLResponse)
async def dong(request: Request):
    # book = BookModel(keyword="파이썬", publisher="BJ퍼플릭", price=1200, image="me.png")
    # print(await mongodb.engine.save(book))
    return templates.TemplateResponse(
        "index.html", {"request": request, "title": " 밥먹러 갑시다."}
    )


@app.get("/shaf", response_class=HTMLResponse)
async def shaf(request: Request):
    # book = BookModel(keyword="파이썬", publisher="BJ퍼플릭", price=1200, image="me.png")
    # print(await mongodb.engine.save(book))
    return templates.TemplateResponse(
        "index.html", {"request": request, "title": " 01011100110."}
    )


@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str):
    # 1. 쿼리에서 검색어 추출
    keyword = q
    # (예외처리)
    # - 검색어가 없다면 사용자에게 검색을 요구 return
    if not keyword:
        return templates.TemplateResponse(
            "./index.html",
            {"request": request, "title": "우리동네 도서관 feat.샤프코드"},
        )

    # - 해당 검색어에 대해 수집된 데이터가 이미 DB에 존재한다면 해당 데이터를 사용자에게 보여준다. return
    if await mongodb.engine.find_one(BookModel, BookModel.keyword == keyword):
        books = await mongodb.engine.find(BookModel, BookModel.keyword == keyword)
        print("이미 존재하는 검색어입니다. DB에 저장된, 기존의 Books를 반환합니다.")
        context = {"request": request, "keyword": keyword, "books": books}

        for book in books:
            print("--------------book 시작---------------------------")
            print(book)

            print("--------------book 끝---------------------------")
            # print(book[1]),

            print("---------------publisher 시작--------------------------")
            print(book.publisher)
            print(book.price)
            print("---------------publisher 끝--------------------------")
        return templates.TemplateResponse("index.html", context=context)

    # 2. 데이터 수집기로 해당 검색어에 대해 데이터를 수집한다.
    naver_book_scraper = NaverBookScraper()
    books = await naver_book_scraper.search(keyword, 10)
    book_models = []

    for book in books:
        print("--------------book 시작---------------------------")
        print(book)
        print("--------------book 끝---------------------------")
        # print(book[1]),
        book_model = BookModel(
            keyword=keyword,
            publisher=book["publisher"],  # book["publisher"],
            price=book["discount"],
            image=book["image"],
        )
        print("---------------book_model 시작--------------------------")
        print(book_model)
        print("---------------book_model 끝--------------------------")

        # 3. DB에 수집된 데이터를 저장한다.
        book_models.append(book_model)
    await mongodb.engine.save_all(book_models)
    # - 수집된 각각의 데이터에 대해 DB에 들어갈 모델 인스턴스를 찍는다.
    # - 각 모델 인스턴스를 DB에 저장한다.

    print(f"입력된 값은 {q} 입니다.")
    return templates.TemplateResponse(
        "./index.html",
        {"request": request, "title": "우리동네 도서관 feat.샤프코드", "books": books},
    )


@app.on_event("startup")
def on_app_start():
    print("Hello Server!")
    """before app starts"""
    mongodb.connect()


@app.on_event("shutdown")
def on_app_shutdown():
    print("Bye Server!")
    """after app shutdown"""
    mongodb.close()


@app.post("/photo")
async def upload_photo(file: UploadFile):
    UPLOAD_DIR = "./photo"  # 이미지를 저장할 서버 경로

    content = await file.read()
    filename = f"{str(time())}.jpg"  # uuid로 유니크한 파일명으로 변경
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)

    return {"filename": filename}


@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):
    contents = await myfile.read()
    return {"filenames": [file.filename for file in files]}


@app.get("/file")
async def main():
    content = """
        <body>
        <form action="/files/" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <input type="submit">
        </form>
        <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <input type="submit">
        </form>
        </body>
    """
    return HTMLResponse(content=content)


# @app.get("/items/{id}", response_class=HTMLResponse)
# async def read_item(request: Request, id: str):
#     return templates.TemplateResponse(
#         "item.html", {"request": request, "id": id, "data": "Hello FastAPI"}
#     )


@app.get("/miri")
def read_miri():
    print(" ")
    return {"미리야": "보고싶어, 사랑해"}


# from typing import Union
# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hellos": "World"}


# @app.get("/miri")
# def read_miri():
#     print(" ")
#     return {"미리야": "사랑해"}


# @app.get("/items/{item_id}/{xyz}")
# def read_item(item_id: int, xyz: str, q: Union[str, None] = None):
#     # code...
#     return {"item_id": item_id, "q": q, "xyz": xyz}
