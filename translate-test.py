# main.py
from main import translate_text

src_lang = "eng_Latn"
tgt_lang = "hin_Deva"
text = "In this story, an angler finds a chest in the Tigris River that he offers to Harun al-Rashid, the Abbasid Caliph. Harun finds that it contains the body of a dead lady and orders his counselor, Ja'far, to tackle it.The dead lady husband and father both claimed to have killed her, however, the Caliph accepts the tale of the spouse who trusted her to have been untrustworthy. The spouse had purchased three exceptional apples for her when she was sick, and when he found a slave with one of the apples, the slave guaranteed his sweetheart gave it to him. In a fury, the man killed his wife. The slave who created all the mischief turns out to be Ja'far's, and Ja'far asks for an exoneration."

translated_text = translate_text(text, src_lang, tgt_lang)
print(f"Translated text: {translated_text}")
