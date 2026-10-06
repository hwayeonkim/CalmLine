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
    r"sons? of a bitch", r"screw (?:you|this|that|off)", r"goddam\w*", r"god damn\w*",
    r"f\w*ck (?:off|you|this|that|it|me)", r"(?:the|a) f\w*ck",
]
MILD = [r"damn\w*", r"hell", r"crap\w*", r"pissed", r"sucks?"]
INSULT = [
    r"idiots?", r"morons?", r"imbeciles?", r"clowns?", r"losers?", r"scum(?:bags?)?", r"parasites?", r"pigs?",
    r"(?:you|ur|you're|youre|your|ya)\s+(?:\w+\s+){0,3}(?:stupid|dumb|useless|worthless|incompetent|pathetic|lazy|brain\s*dead|garbage|trash|joke)",
    r"(?:stupid|dumb|useless|worthless|incompetent|pathetic|lazy)\s+(?:\w+\s+){0,2}(?:people|bureaucrats?|workers?|operators?|lady|woman|man|girl|boy|clerks?|agents?)",
    r"shut up", r"shut your (?:mouth|face|trap)", r"get a brain", r"retard\w*", r"half[- ]?wit", r"waste of (?:space|air|oxygen|a salary|my taxes)",
    r"do you even (?:speak|understand|have a brain)", r"are you (?:deaf|stupid|dumb|an idiot|slow|retarded)",
    r"jerks?", r"twits?", r"(?:the )?dumbest \w+(?: \w+)?", r"who (?:the (?:hell|fuck|heck) )?(?:trained|hired|raised) you",
]
SEXUAL = [
    r"sweet(?:ie|heart|cheeks)", r"cutie(?: pie)?", r"\bhoney\b", r"\bdarling\b", r"\bbabe\b", r"hey baby", r"\bsexy\b", r"what (?:are|r) you wearing", r"your (?:body|legs|boobs|tits|ass|butt|lips)",
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

CATEGORIES = {
    "profanity": PROFANITY,
    "mild": MILD,
    "insult": INSULT,
    "sexual": SEXUAL,
    "identity": IDENTITY,
    "threat": THREAT,
    "self_harm": SELF_HARM,
}

COMPILED = {k: [re.compile(r"(?<![\w])" + p + r"(?![\w])", re.I) for p in v] for k, v in CATEGORIES.items()}


def find_spans(text: str, include_mild: bool = True):
    out = []
    for cat, pats in COMPILED.items():
        if cat == "mild" and not include_mild:
            continue
        for p in pats:
            for m in p.finditer(text):
                out.append((m.start(), m.end(), cat, m.group(0)))
    out.sort()
    return out
