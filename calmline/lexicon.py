"""Hand-built abuse lexicon (v0 baseline).

Patterns are regexes applied case-insensitively with word boundaries.
Severity: mild profanity (damn, hell, crap) is tracked separately so policies can
choose whether to filter it.
"""
import re

PROFANITY = [
    r"f+u+c+k+\w*", r"f[\*\#@]+c?k\w*", r"f\*+\w*", r"motherf\w*", r"effing", r"frigging", r"freaking",
    r"sh[i\*]t+\w*", r"bullsh\w*", r"b[i\*]tch\w*", r"bastards?", r"assholes?", r"a\*\*hole\w*", r"jackass\w*", r"dumbass\w*",
    r"dick(?:head)?s?", r"cunts?", r"pricks?", r"douche\w*", r"wtf", r"stfu", r"pissed off", r"piss off",
    r"sons? of a bitch", r"screw you", r"goddamn\w*", r"god damn\w*",
]
MILD = [r"damn\w*", r"hell", r"crap\w*", r"pissed", r"sucks?"]
INSULT = [
    r"idiots?", r"morons?", r"imbeciles?", r"clowns?", r"losers?", r"scum(?:bags?)?", r"parasites?", r"pigs?",
    r"(?:you|ur|you're|youre|your|ya)\s+(?:\w+\s+){0,3}(?:stupid|dumb|useless|worthless|incompetent|pathetic|lazy|brain\s*dead|garbage|trash|joke)",
    r"(?:stupid|dumb|useless|worthless|incompetent|pathetic|lazy)\s+(?:\w+\s+){0,2}(?:people|bureaucrats?|workers?|operators?|lady|woman|man|girl|boy|clerks?|agents?)",
    r"shut up", r"shut your (?:mouth|face|trap)", r"get a brain", r"retard\w*", r"half[- ]?wit", r"waste of (?:space|air|oxygen|a salary|my taxes)",
    r"do you even (?:speak|understand|have a brain)", r"are you (?:deaf|stupid|dumb|an idiot|slow|retarded)",
]
SEXUAL = [
    r"sweet(?:ie|heart|cheeks)", r"\bbabe\b", r"hey baby", r"\bsexy\b", r"what (?:are|r) you wearing", r"your (?:body|legs|boobs|tits|ass|butt|lips)",
    r"suck my", r"blow ?job", r"\bnaked\b", r"\bhorny\b", r"sit on my", r"come (?:over|to my place) (?:and|to)", r"in (?:my|your) bed",
    r"turn(?:s|ing)? me on", r"i(?:'d| would) (?:love to )?(?:do|bang|screw|f\w+) you", r"hot (?:voice|lady|girl)", r"talk dirty",
]
IDENTITY = [
    r"go back to (?:your|ur) (?:own )?country", r"speak (?:proper |real )?english", r"(?:diversity|affirmative action) hire",
    r"your kind", r"you foreigners?", r"(?:dumb|stupid) (?:immigrants?|foreigners?)",
]
THREAT = [
    r"(?:i'?ll|i will|i'm gonna|im gonna|i am going to|gonna|going to)\s+(?:\w+\s+){0,2}(?:kill|shoot|stab|hurt|beat|strangle|smash|break|burn|end|find|hunt|get)\b",
    r"(?:kill|shoot|stab|strangle) (?:you|him|her|them|everyone|somebody|someone)",
    r"(?:know|found out) where you (?:work|live|sit)", r"watch your back", r"you(?:'ll| will) (?:regret|pay|be sorry)",
    r"(?:come|coming) down there", r"wait(?:ing)? (?:for you )?outside", r"\bbomb\b", r"\bmy gun\b", r"bring (?:my|a) (?:gun|knife|bat)",
    r"burn (?:the|your|this) (?:place|building|office)", r"blow (?:the|this|your) (?:place|office|building) up", r"make you pay",
    r"see what happens", r"you(?:'re| are) dead", r"\bdead meat\b",
]
SELF_HARM = [
    r"kill myself", r"end (?:it all|my life)", r"(?:don't|do not|dont) want to (?:live|be alive)", r"jump off", r"suicid\w*", r"hurt myself",
]

# v3 additions (from dev-set error analysis, 2026-10-06)
EXTENSIONS = {'profanity': ['screw (?:you|this|that|off)', 'goddam\\w*', 'f\\w*ck (?:off|you|this|that|it|me)', '(?:the|a) f\\w*ck'], 'insult': ['jerks?', 'twits?', '(?:the )?dumbest \\w+(?: \\w+)?', 'who (?:the (?:hell|fuck|heck) )?(?:trained|hired|raised) you'], 'sexual': ['cutie(?: pie)?', '\\bhoney\\b', '\\bdarling\\b']}

CATEGORIES = {
    "profanity": PROFANITY,
    "mild": MILD,
    "insult": INSULT,
    "sexual": SEXUAL,
    "identity": IDENTITY,
    "threat": THREAT,
    "self_harm": SELF_HARM,
}


def _compile(cats):
    return {k: [re.compile(r"(?<![\w])" + p + r"(?![\w])", re.I) for p in v] for k, v in cats.items()}


# v4 additions (from synthetic-dev error analysis): generalised threat / self-harm / identity forms
EXTENSIONS_V4 = {
    "mild": [r"bloody"],
    "threat": [r"(?:put|pump) (?:a )?(?:bullet|bullets|hole) in", r"(?:find|get|wait for|follow) you after (?:work|your shift)",
               r"bring(?:ing)? (?:a|my) (?:bat|gun|knife|weapon)", r"smash your (?:face|head|skull)", r"(?:beat|kick) the (?:crap|shit|hell) out of",
               r"(?:know|found out) where (?:you|your \w+) (?:work|works|live|lives|is|are)", r"(?:burn|blow) (?:\w+ ){0,2}(?:down|up)\b"],
    "self_harm": [r"end(?:ing)? (?:it all|my life|myself)", r"(?:want|wanna|going) to die", r"(?:no|not any) (?:reason|point) (?:to|in) (?:live|living)"],
    "identity": [r"(?:put|get) (?:someone|somebody) on who speaks english", r"go back (?:to )?where you came from",
                 r"(?:want|need) (?:a )?real american", r"your kind"],
    "sexual": [r"(?:sit|sitting) on my (?:lap|face)", r"(?:you )?(?:sound|look|are) (?:so |really |very )?(?:sexy|hot|cute)", r"i bet you(?:'re| are)? ?(?:look )?(?:hot|sexy|naked)"],
    "insult": [r"(?:get|pull) your head out of your ass", r"are you even listening", r"num(?:b)?skulls?", r"halfwits?"],
}

# v7: fix over-broad self-harm pattern found on Gemma-written hard negatives
#     ("I'm worried someone is going to die" is concern, not self-harm) -> require a first-person subject.
V7_REMOVE = {"self_harm": [r"(?:want|wanna|going) to die"]}
V7_ADD = {"self_harm": [r"(?:i|i'm|im|i am)\s+(?:\w+\s+)?(?:want|wanna|going|gonna|ready) to die", r"\bi want to die\b"]}

COMPILED_V0 = _compile(CATEGORIES)
COMPILED_V3 = _compile({k: v + EXTENSIONS.get(k, []) for k, v in CATEGORIES.items()})
COMPILED = _compile({k: v + EXTENSIONS.get(k, []) + EXTENSIONS_V4.get(k, []) for k, v in CATEGORIES.items()})


COMPILED_V7 = _compile({k: [p for p in v + EXTENSIONS.get(k, []) + EXTENSIONS_V4.get(k, []) if p not in V7_REMOVE.get(k, [])] + V7_ADD.get(k, [])
                        for k, v in CATEGORIES.items()})


def find_spans(text: str, include_mild: bool = True, extended=True):
    """extended: False -> v0 lexicon, True/'v3' -> v3 lexicon, 'v4' -> v4 lexicon, 'v7' -> v7 lexicon."""
    table = {False: COMPILED_V0, True: COMPILED_V3, "v3": COMPILED_V3, "v4": COMPILED, "v7": COMPILED_V7}[extended]
    out = []
    for cat, pats in table.items():
        if cat == "mild" and not include_mild:
            continue
        for p in pats:
            for m in p.finditer(text):
                out.append((m.start(), m.end(), cat, m.group(0)))
    out.sort()
    return out
