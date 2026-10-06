"""Rule-based slot extraction. Slots are *extracted verbatim* from the caller's words
(never generated) so they can be trusted when re-rendered in a calm template."""
import re

SUFFIX = (r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|place|pl|drive|dr|lane|ln|parkway|pkwy|court|ct|terrace|"
          r"highway|hwy|concourse|plaza|square|expressway|way|loop|walk|park)")
NAME = r"(?:(?!(?:on|at|the|in|to|of|and|near|by|from|is|a|an|my|our|its|it's|has|was|off)\b)[A-Za-z][\w']*\.?)"
STREET = rf"(?:(?:avenue|ave) [a-z]\b|broadway|(?:{NAME} ){{0,3}}?\d+(?:st|nd|rd|th)? {SUFFIX}\b|(?:{NAME} ){{1,3}}{SUFFIX}\b|bay \d+(?:st|nd|rd|th))"
HOUSE = r"\d+(?:-\d+)?"

PATTERNS = [
    ("reference", rf"\b311-\d{{6,10}}\b|\b(?:reference|ref|ticket|case|complaint|confirmation|sr)\s*(?:number|no\.?|#)?\s*(?:is\s*)?[:#]?\s*([A-Z0-9][A-Z0-9-]{{5,}})"),
    ("address", rf"\b{HOUSE}\s+(?:(?:west|east|north|south|w|e|n|s)\.?\s+)?{STREET}"),
    ("cross_street", rf"\b(?:corner of |at |on |near )?(?:{STREET}|\d+(?:st|nd|rd|th)|(?-i:[A-Z][a-z]+))\s+(?:and|&)\s+(?:{STREET}|\d+(?:st|nd|rd|th)(?: {SUFFIX})?|(?-i:[A-Z][a-z]+)(?: {SUFFIX})?)"),
    ("street", rf"\b(?:on|at|near|off|along|down|of|by|to|onto|in)\s+(?:the\s+)?({STREET})"),
    ("apartment", r"\b(?:apartment|apt\.?|unit)\s*#?\s*([0-9]{1,2}[a-z]?|[a-z][0-9]{0,2})\b|\b(?:in|from|at|,)\s+([0-9]{1,2}[A-H])\b|\b([0-9]{1,2}[A-H])\b(?=,? )"),
    ("floor", r"\b(?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|fourteenth)\s+floor\b"),
    ("borough", r"\b(?:brooklyn|queens|the bronx|bronx|manhattan|staten island|astoria|harlem)\b"),
    ("plate", r"\b(?:plate|license)\s*(?:number|#)?\s*(?:is\s*)?([A-Z]{2,4}[ -]?\d{3,4}|\d{3,4}[ -]?[A-Z]{2,4})\b|\b[A-Z]{3} \d{4}\b"),
    ("vehicle", r"\b(?:a|an|the|this|that)\s+((?:(?:black|white|gray|grey|silver|red|blue|green|yellow|brown|old|commercial|burned out)\s+)*(?:honda|toyota|nissan|bmw|ford|chevy|jeep|tesla|suv|van|truck|pickup truck|box truck|sedan|car|motorcycle|bus))\b"),
    ("time", r"\b(?:since|from|until|till|by|after|before|around|at|starting at)\s+(?:like\s+)?(?:\d{1,2}(?::\d\d)?\s*(?:am|pm|a\.m\.|p\.m\.|in the (?:morning|evening|afternoon))?|noon|midnight|yesterday(?: morning| afternoon| evening)?|last (?:night|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|this (?:morning|afternoon|week)|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?|january|february|march|april|may|june|july|august|september|october|november|december)\b"
             r"|\b\d{1,2}(?::\d\d)?\s*(?:am|pm|a\.m\.|p\.m\.)\b|\b\d{1,2} in the (?:morning|evening|afternoon)\b|\b(?:every (?:single )?(?:night|day|morning|weekend)|all (?:night|day|afternoon)|tonight|this morning|last night|yesterday)\b|\bon (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b"),
    ("duration", r"\b(?:for|in|over)\s+(?:like\s+|almost\s+|about\s+|over\s+)?(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|several|a few|many|forty)\s+(?:minutes?|hours?|days?|weeks?|months?|years?)(?: straight| now)?\b|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:minutes?|hours?|days?|weeks?|months?|years?)\b(?: ago)?|\b(?:a|one) (?:week|month|year)\b"),
    ("prior_contact", r"\b(?:called|reported|complained|calling)\b(?:\s+\w+){0,4}?\s+(?:\d+|two|three|four|five|six|seven|eight|nine|ten|several|many|multiple|a dozen)\s+times\b|\b(?:this is (?:my |the )?)?(?:second|third|fourth|fifth|sixth|seventh|\d+(?:st|nd|rd|th))(?: \w+)? (?:time|call)\b|\bcall number \w+\b|\b(?:\w+ )?times already\b|\balready (?:called|reported|complained)\b|\bon hold for (?:\w+ ){1,2}(?:minutes|hours?)\b|\bholding for (?:\w+ ){1,2}minutes\b"),
    ("vulnerable", r"\b(?:my )?(?:\d+|two|three|four|five|six)?\s*(?:elderly|sick|disabled|pregnant|newborn|asthma\w*)\b(?:\s+\w+)?|\b(?:my|three|four|\d+|the) (?:kids?|baby|children|grandma|mother|son|daughter|wife)\b(?:'s)?|\bwheelchair\b|\b\d+ years old\b|\bdialysis\b|\bknee surgery\b|\bnurse\b|\bhas asthma\b|\b\d+ people\b"),
    ("request", r"\b(?:i want|i need|i'd like|can you|could you|please|can someone|can somebody|send|get|need someone to|somebody needs to|someone needs to)\b\s+(?:\w+\s+){0,2}?(?:a |an |the )?(?:supervisor|inspector|police|cops|someone|somebody|tow(?: truck)?|exterminator|con ed|parks people|an? email)(?:\s+(?:to|out|there|here|over)\b(?:\s+\w+){0,4})?|\bcall me back\b|\b(?:tow|towed) (?:it|the \w+)\b|\bcheck on (?:him|her|them)\b"),
]
COMPILED = [(n, re.compile(p, re.I)) for n, p in PATTERNS]


def extract(text: str) -> dict:
    """Return {slot_type: [verbatim strings]} with overlaps resolved (longer spans win)."""
    cands = []
    for name, pat in COMPILED:
        for m in pat.finditer(text):
            s, e = m.span()
            val = text[s:e].strip(" ,.")
            if name == "street":
                s, e = m.span(1)
                val = m.group(1)
            if len(val) < 2:
                continue
            cands.append((s, e, name, val))
    # resolve overlaps of location-like slots: prefer address > cross_street > street
    prio = {"reference": 0, "address": 1, "cross_street": 2, "plate": 2, "street": 3}
    cands.sort(key=lambda c: (prio.get(c[2], 5), -(c[1] - c[0])))
    taken, out = [], {}
    for s, e, name, val in cands:
        if any(not (e <= ts or s >= te) for ts, te in taken):
            continue
        taken.append((s, e))
        out.setdefault(name, []).append((s, val))
    return {k: [v for _, v in sorted(vs)] for k, vs in out.items()}


def spans(text: str):
    """Character spans of all extracted slots (used to protect facts during masking)."""
    res = []
    for name, pat in COMPILED:
        for m in pat.finditer(text):
            res.append((m.start(), m.end(), name))
    return res
