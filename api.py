from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

@app.post("/translate/")
async def translate_text(req: TranslateRequest):
    from main import translate_text
    return {"translated_text": translate_text(req.text, req.src_lang, req.tgt_lang)}