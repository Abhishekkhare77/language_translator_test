from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer.IndicTransTokenizer import IndicTransTokenizer, IndicProcessor

app = FastAPI()

class TranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_QUANTIZATION = None  # Set to "8-bit" if you want to use 8-bit quantization by default

def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
    if DEVICE == "cuda" and quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    model = model.to(DEVICE)
    if DEVICE == "cuda" and qconfig is None:
        model.half()

    model.eval()
    return tokenizer, model

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        inputs = tokenizer(
            batch,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

    return translations

@app.on_event("startup")
async def startup_event():
    global en_indic_tokenizer, en_indic_model, indic_en_tokenizer, indic_en_model, indic_indic_tokenizer, indic_indic_model
    en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
    indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"
    indic_indic_ckpt_dir = "ai4bharat/indictrans2-indic-indic-1B"

    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", DEFAULT_QUANTIZATION)
    indic_en_tokenizer, indic_en_model = initialize_model_and_tokenizer(indic_en_ckpt_dir, "indic-en", DEFAULT_QUANTIZATION)
    indic_indic_tokenizer, indic_indic_model = initialize_model_and_tokenizer(indic_indic_ckpt_dir, "indic-indic", DEFAULT_QUANTIZATION)

@app.post("/translate/")
async def translate_text(req: TranslateRequest):
    if req.src_lang.startswith("eng") and req.tgt_lang.startswith("indic"):
        tokenizer, model = en_indic_tokenizer, en_indic_model
    elif req.src_lang.startswith("indic") and req.tgt_lang.startswith("eng"):
        tokenizer, model = indic_en_tokenizer, indic_en_model
    else:
        tokenizer, model = indic_indic_tokenizer, indic_indic_model

    ip = IndicProcessor(inference=True)
    translations = batch_translate([req.text], req.src_lang, req.tgt_lang, model, tokenizer, ip)
    return {"translated_text": translations[0]}
