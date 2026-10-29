"""The languages text can be read in: Tesseract's codes, their names, and which are printed on packs most often.

Several are read at once by joining their codes with "+" ("eng+ara"), the first given first priority. Each one added
makes reading slower and gives the reader more ways to misread, so only the languages actually printed are chosen.
"""

from __future__ import annotations

DEFAULT_LANGUAGES: tuple[str, ...] = ("eng",)
AUTO = "auto"  # the languages found on the files themselves, while they are compared (`app.engine.language_detect`)

# Offered first, as tick boxes: what pharmaceutical and food packs are most often printed in.

LANGUAGE_NAMES: dict[str, str] = {
    "afr": "Afrikaans", "amh": "Amharic", "ara": "Arabic", "asm": "Assamese", "aze": "Azerbaijani",
    "aze_cyrl": "Azerbaijani (Cyrillic)", "bel": "Belarusian", "ben": "Bengali", "bod": "Tibetan", "bos": "Bosnian",
    "bre": "Breton", "bul": "Bulgarian", "cat": "Catalan", "ceb": "Cebuano", "ces": "Czech",
    "chi_sim": "Chinese (Simplified)", "chi_sim_vert": "Chinese (Simplified, vertical)", "chi_tra": "Chinese (Traditional)",
    "chi_tra_vert": "Chinese (Traditional, vertical)", "chr": "Cherokee", "cos": "Corsican", "cym": "Welsh",
    "dan": "Danish", "deu": "German", "deu_latf": "German (Fraktur)", "div": "Dhivehi", "dzo": "Dzongkha",
    "ell": "Greek", "eng": "English", "enm": "English (Middle)", "epo": "Esperanto", "equ": "Equations",
    "est": "Estonian", "eus": "Basque", "fao": "Faroese", "fas": "Persian", "fil": "Filipino", "fin": "Finnish",
    "fra": "French", "frk": "Frankish", "frm": "French (Middle)", "fry": "Frisian", "gla": "Scottish Gaelic",
    "gle": "Irish", "glg": "Galician", "grc": "Greek (Ancient)", "guj": "Gujarati", "hat": "Haitian Creole",
    "heb": "Hebrew", "hin": "Hindi", "hrv": "Croatian", "hun": "Hungarian", "hye": "Armenian", "iku": "Inuktitut",
    "ind": "Indonesian", "isl": "Icelandic", "ita": "Italian", "ita_old": "Italian (Old)", "jav": "Javanese",
    "jpn": "Japanese", "jpn_vert": "Japanese (vertical)", "kan": "Kannada", "kat": "Georgian", "kat_old": "Georgian (Old)",
    "kaz": "Kazakh", "khm": "Khmer", "kir": "Kyrgyz", "kmr": "Kurdish (Kurmanji)", "kor": "Korean",
    "kor_vert": "Korean (vertical)", "lao": "Lao", "lat": "Latin", "lav": "Latvian", "lit": "Lithuanian",
    "ltz": "Luxembourgish", "mal": "Malayalam", "mar": "Marathi", "mkd": "Macedonian", "mlt": "Maltese",
    "mon": "Mongolian", "mri": "Maori", "msa": "Malay", "mya": "Burmese", "nep": "Nepali", "nld": "Dutch",
    "nor": "Norwegian", "oci": "Occitan", "ori": "Odia", "pan": "Punjabi", "pol": "Polish", "por": "Portuguese",
    "pus": "Pashto", "que": "Quechua", "ron": "Romanian", "rus": "Russian", "san": "Sanskrit", "sin": "Sinhala",
    "slk": "Slovak", "slv": "Slovenian", "snd": "Sindhi", "spa": "Spanish", "spa_old": "Spanish (Old)",
    "sqi": "Albanian", "srp": "Serbian", "srp_latn": "Serbian (Latin)", "sun": "Sundanese", "swa": "Swahili",
    "swe": "Swedish", "syr": "Syriac", "tam": "Tamil", "tat": "Tatar", "tel": "Telugu", "tgk": "Tajik", "tha": "Thai",
    "tir": "Tigrinya", "ton": "Tongan", "tur": "Turkish", "uig": "Uyghur", "ukr": "Ukrainian", "urd": "Urdu",
    "uzb": "Uzbek", "uzb_cyrl": "Uzbek (Cyrillic)", "vie": "Vietnamese", "yid": "Yiddish", "yor": "Yoruba",
}

# Tesseract's own data files, not languages a pack is printed in.


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def languages_text(codes: tuple[str, ...] | list[str]) -> str:
    """A choice of languages as a reviewer reads it: "English, Arabic and French"."""
    names = [language_name(code) for code in codes]
    if len(names) <= 1:
        return names[0] if names else language_name(DEFAULT_LANGUAGES[0])
    return ", ".join(names[:-1]) + " and " + names[-1]


def joined(codes: tuple[str, ...] | list[str]) -> str:
    """The choice as Tesseract takes it ("eng+ara"): English first when chosen (its model is the most reliable reader
    of the numbers and Latin text every pack carries), then the rest in the order given."""
    ordered = sorted(codes, key=lambda code: code != "eng")
    return "+".join(ordered) if ordered else "+".join(DEFAULT_LANGUAGES)
