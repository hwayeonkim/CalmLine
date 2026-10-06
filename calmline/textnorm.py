"""Normalisation used for fact matching and leak checks (evaluation + verification)."""
import re

NUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
       "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
       "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "ninety": 90, "hundred": 100}
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
       "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "once": 1, "twice": 2, "thrice": 3}
ABBR = {"ave": "avenue", "st": "street", "blvd": "boulevard", "apt": "apartment", "pkwy": "parkway", "rd": "road", "dr": "drive",
        "pl": "place", "ln": "lane", "hwy": "highway", "a.m.": "am", "p.m.": "pm", "mins": "minute", "hrs": "hour", "w": "west",
        "e": "east", "n": "north", "s": "south", "cops": "police", "unit": "apartment"}
STOP = {"the", "a", "an", "of", "on", "in", "at", "to", "my", "and", "is", "it", "for", "from", "with", "this", "that", "i",
        "o'clock", "oclock", "about", "around", "like", "has", "have", "been", "every", "single"}


def tokens(text: str):
    t = text.lower().replace("’", "'")
    t = re.sub(r"(\d)(am|pm|a\.m\.|p\.m\.)\b", r"\1 \2", t)
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"(\d+):00\b", r"\1", t)
    out = []
    for w in re.findall(r"[a-z0-9]+(?:[-'.][a-z0-9]+)*", t):
        w = w.strip(".'")
        w = ABBR.get(w, w)
        if w in NUM:
            w = str(NUM[w])
        elif w in ORD:
            w = str(ORD[w])
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and not w[:-1].isdigit():
            w = w[:-1]
        if w.endswith("'s"):
            w = w[:-2]
        out.append(w)
    return out


def content_tokens(text: str):
    return [w for w in tokens(text) if w not in STOP]


def fact_present(fact: str, delivered: str) -> bool:
    ft = content_tokens(fact)
    if not ft:
        return True
    dt = set(tokens(delivered))
    hit = sum(1 for w in ft if w in dt)
    need = len(ft) if len(ft) <= 3 else int(0.75 * len(ft) + 0.999)
    return hit >= need


LEAK_STOP = STOP | {"you", "your", "you're", "are", "people", "so", "be", "do", "me", "what", "get", "just", "all", "out",
                    "whole", "down", "up", "over", "him", "her", "see", "come", "going", "gonna", "i'll", "i'm", "it'll",
                    "maybe", "lady", "after", "can", "or", "sake", "this", "here", "there", "too", "hard", "even", "operator",
                    "ever", "talked", "next", "time", "building", "office", "who", "which", "kind", "own", "country"}


def leaked(span: str, delivered: str) -> bool:
    key = [w for w in tokens(span) if w not in LEAK_STOP]
    if not key:
        key = tokens(span)
    d = " " + " ".join(tokens(delivered)) + " "
    if " " + " ".join(tokens(span)) + " " in d:
        return True
    dt = set(d.split())
    hit = sum(1 for w in key if w in dt)
    return hit >= max(1, (len(key) + 1) // 2)


def numbers(text: str):
    return set(re.findall(r"\d+(?:-\d+)?", " ".join(tokens(text))))
