from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rokosms.predict import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    text: str


@app.on_event("startup")
def warm_up():
    predict("warming up the model")


@app.get("/")
def health_check():
    return {"status": "RokoSMS API is running"}


@app.post("/predict")
def predict_message(message: Message):
    return predict(message.text)
