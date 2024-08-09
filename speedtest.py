from fastapi import FastAPI,HTTPException       
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer.IndicTransTokenizer import IndicTransTokenizer, IndicProcessor
import httpx
import threading
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import asyncio


# MongoDB connection URI
uri = "mongodb+srv://concur-admin:oApIL0eGKTzHeWrn@concur-backend-db.3jzk7uh.mongodb.net"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Access the specific database
db = client.Consent_directory_db

# Access the specific collection
languages = db["consent_language"]
industries = db["consent_industry"]
purposes = db["consent_directory"]

new_purposes_collection = db["consent_directory_new"]
new_purposes_collection2 = db["consent_directory_new_krishna"]
new_translated_collection = db["consent_directory_translated"]

purposes.create_index([("$**", "text")])



# Initialize APScheduler
scheduler = BackgroundScheduler()


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

    scheduler.start()


    
async def translate_text(text: str, src_lang: str, tgt_lang: str):
    print("Translating text:", text, "from", src_lang, "to", tgt_lang)
    if src_lang.startswith("eng") and tgt_lang.startswith("indic"):
        tokenizer, model = en_indic_tokenizer, en_indic_model
    elif src_lang.startswith("indic") and tgt_lang.startswith("eng"):
        tokenizer, model = indic_en_tokenizer, indic_en_model
    else:
        tokenizer, model = indic_indic_tokenizer, indic_indic_model

    ip = IndicProcessor(inference=True)
    translations = batch_translate([text], src_lang, tgt_lang, model, tokenizer, ip)
    print("Translation successful. For", src_lang, "to", tgt_lang)
    return {"translated_text": translations[0]}

async def translate_purposes():
    try:
        all_languages = list(languages.find())
        documents = list(new_purposes_collection2.find({"is_translated": {"$ne": True}}))

        for doc in documents:
            print("Processing document ID", doc["_id"])
            if not doc.get("is_translated"):
                english_purpose = None
                for p in doc.get("purpose", []):
                    if p["lang_short_code"] == "en":
                        english_purpose = p["description"]
                        break

                if not english_purpose:
                    print(f"No English purpose found for document ID {doc['_id']}. Skipping translation.")
                    continue

                for purpose in doc.get("purpose", []):
                    if not purpose["description"]:  # Only translate if description is empty
                        for lang in all_languages:
                            if purpose["lang_short_code"] == lang["lang_short_code"]:
                                translated_text = await translate_text(
                                    english_purpose,
                                    "en_Latn",
                                    lang['translation_symbol']
                                )
                                purpose["description"] = translated_text["translated_text"]
                                print(f"Updating document ID {doc['_id']} for language {lang['lang_short_code']}.")
                                new_purposes_collection2.update_one(
                                    {"_id": doc["_id"], "purpose.lang_short_code": purpose["lang_short_code"]},
                                    {
                                        "$set": {
                                            "purpose.$.description": purpose["description"],
                                            "updated_at": datetime.now(),
                                        }
                                    },
                                )

                if all(p.get("description") for p in doc.get("purpose", [])):
                    print(f"Marking document ID {doc['_id']} as fully translated.")
                    new_purposes_collection2.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {"is_translated": True}}
                    )

        print("Language translations updated successfully.")
    except Exception as e:
        print(f"An error occurred during language translation: {e}")


def run_translate_purposes():
    asyncio.run(translate_purposes())

# scheduler.add_job(run_translate_purposes, IntervalTrigger(seconds=30))
scheduler.add_job(run_translate_purposes, "cron", hour=18, minute=44, second=00)

@app.post("/trigger-translation/")
async def translate_all_purposes():
    threading.Thread(target=run_translate_purposes).start()
    return {"message": "Translation process initiated."}



# async def text_translation(text : str, src_lang : str, tgt_lang : str):
#     print("Entered translation function with text:", text, "from", src_lang, "to", tgt_lang)
#     async with httpx.AsyncClient() as client:
#         payload = {
#             "text": text,
#             "src_lang": src_lang,
#             "tgt_lang": tgt_lang
#         }
#         try:
#             response = await client.post(
#                 'https://translator.adnan-qasim.me/translate/',
#                 json=payload,
#                 headers={'accept': 'application/json', 'Content-Type': 'application/json'}
#             )
#             response.raise_for_status()  # Raise an error for 4xx/5xx responses
#         except httpx.RequestError as exc:
#             print(f"An error occurred while requesting {exc.request.url!r}.")
#             raise HTTPException(status_code=500, detail="Translation service request failed")
#         except httpx.HTTPStatusError as exc:
#             print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
#             raise HTTPException(status_code=500, detail="Translation service failed")

#         translated_text = response.json().get("translated_text")
#         if translated_text:
#             print("Translation successful. For", src_lang, "to", tgt_lang)
#             return translated_text
#         else:
#             print("Translation failed for", src_lang, "to", tgt_lang)
#             raise HTTPException(status_code=500, detail="Received empty translation")