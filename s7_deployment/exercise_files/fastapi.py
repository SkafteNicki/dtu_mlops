from __future__ import annotations

import re
from enum import Enum
from http import HTTPStatus

import anyio
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
def read_root():
    """Simple root endpoint."""
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    """Simple function to get an item by id."""
    return {"item_id": item_id}


class ItemEnum(Enum):
    """Item enum."""

    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):  # noqa: F811
    """Simple function to get an item by id."""
    return {"item_id": item_id}


@app.get("/query_items")
def read_item(item_id: int):  # noqa: F811
    """Simple function to get an item by id."""
    return {"item_id": item_id}


database = {"username": [], "password": []}


@app.post("/login/")
def login(username: str, password: str) -> str:
    """Simple function to save a login."""
    username_db = database["username"]
    password_db = database["password"]
    if username not in username_db and password not in password_db:
        with open("database.csv", "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"


@app.get("/text_model/")
def contains_email(data: str):
    """Simple function to check if an email is valid."""
    regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data) is not None,
    }


class DomainEnum(Enum):
    """Domain enum."""

    gmail = "gmail"
    hotmail = "hotmail"


class Item(BaseModel):
    """Item model."""

    email: str
    domain: DomainEnum


@app.post("/text_model/")
def contains_email_domain(data: Item):
    """Simple function to check if an email is valid."""
    if data.domain is DomainEnum.gmail:
        regex = r"\b[A-Za-z0-9._%+-]+@gmail+\.[A-Z|a-z]{2,}\b"
    if data.domain is DomainEnum.hotmail:
        regex = r"\b[A-Za-z0-9._%+-]+@hotmail+\.[A-Z|a-z]{2,}\b"
    return {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data.email) is not None,
    }


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: None | int = 28, w: None | int = 28):
    """Simple function using open-cv to resize an image."""
    async with await anyio.open_file("image.jpg", "wb") as image:
        content = await data.read()
        image.write(content)
        image.close()

    img = cv2.imread("image.jpg")
    res = cv2.resize(img, (h, w))

    cv2.imwrite("image_resize.jpg", res)

    return {
        "input": data,
        "output": FileResponse("image_resize.jpg"),
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
