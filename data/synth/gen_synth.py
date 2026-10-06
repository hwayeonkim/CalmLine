"""Compositional generator for a large synthetic DEV set (tuning only — never used for final numbers).

Each utterance = issue clause (intent-specific, with key nouns marked [[...]] as facts)
               + location/time/prior-contact/request/vulnerability slots
               + 0..2 abuse insertions (prefix / inline modifier / suffix / rhetorical / threat)
               + optional figurative-but-clean venting.
Gold facts and toxic spans are recorded at generation time, so labels are exact."""
import json, random, re, sys

R = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 7)

STREETS = ["Nostrand Avenue", "Fulton Street", "Amsterdam Avenue", "Grand Street", "Linden Boulevard", "Steinway Street",
           "Westchester Avenue", "Hylan Boulevard", "Myrtle Avenue", "Kings Highway", "Broadway", "Lexington Avenue",
           "Ocean Avenue", "Jerome Avenue", "Northern Boulevard", "Utica Avenue", "Richmond Avenue", "Atlantic Avenue",
           "Eastern Parkway", "Roosevelt Avenue", "Fordham Road", "Flatbush Avenue", "Court Street", "Bedford Avenue",
           "Tremont Avenue", "Victory Boulevard", "Liberty Avenue", "Church Avenue", "St Johns Place", "Dekalb Avenue",
           "Malcolm X Boulevard", "Pelham Parkway", "Forest Avenue", "Union Turnpike", "Hillside Avenue", "Gun Hill Road"]
ORD_STREETS = ["{n}th Street", "{n}th Avenue", "West {n}th Street", "East {n}th Street"]


def street():
    if R.random() < 0.25:
        n = R.choice([4, 5, 6, 7, 8, 9, 11, 14, 17, 18, 19, 24, 34, 42, 57, 86, 96, 104, 116, 145, 163, 181])
        return R.choice(ORD_STREETS).format(n=n)
    return R.choice(STREETS)


def address():
    hn = R.choice([str(R.randint(1, 2999)), f"{R.randint(10, 199)}-{R.randint(10, 99)}"])
    return f"{hn} {street()}"


def apt():
    return f"{R.randint(1, 18)}{R.choice('ABCDEFGHJKLMNPRS')}"


TIMES = ["since last night", "since Monday", "since Tuesday morning", "since 9am", "at 2am", "around 11pm", "every night",
         "every weekend", "since yesterday", "since the weekend", "since 6 in the morning", "every morning", "all night"]
DURS = ["for three days", "for two weeks", "for a month", "for five hours", "for over a week", "for six months", "for 45 minutes"]
PRIOR = ["I called {k} times already", "this is my {o} call about this", "I already reported it {k} times", "I've complained {k} times"]
KS = ["two", "three", "four", "five", "six", "7", "10"]
ORDS = ["second", "third", "fourth", "fifth"]
REQ = {"I need someone to come out today": "someone today", "please send an inspector": "inspector",
       "I want a supervisor to call me back": "supervisor call back", "can you send someone tonight": "someone tonight",
       "I need the police to come": "police", "please get a tow truck here": "tow truck", "can someone check on him": "check on him"}
VULN = ["my mother is 90 years old", "I have a newborn baby", "my son has asthma", "my husband is in a wheelchair",
        "I'm seven months pregnant", "my daughter has a fever", "I'm a diabetic"]

# intent -> list of (clause template, facts). {L} = location, [[x]] = fact
ISSUES = {
    "illegal_parking": ["a [[delivery truck]] is double parked at {L} blocking the whole lane", "somebody parked a [[gray minivan]] right on the [[hydrant]] at {L}",
                        "cars are parked on the [[crosswalk]] at {L} so strollers can't cross", "a [[tour bus]] parks in the [[bus stop]] at {L} every day"],
    "blocked_driveway": ["a [[blue sedan]] is blocking my [[driveway]] at {L}", "I can't get out of my [[driveway]] at {L}, a [[pickup truck]] is parked across it"],
    "noise_residential": ["the tenant above me at {L} is [[hammering]] on the floor", "my neighbor at {L} has a [[party]] with a [[subwoofer]]",
                          "the [[dog]] next door at {L} barks for hours", "someone in [[apartment {A}]] at {L} plays [[drums]]"],
    "noise_street": ["a crowd outside {L} is [[yelling]] and playing [[music]] from a car", "street [[vendors]] at {L} use [[megaphones]]"],
    "noise_vehicle": ["a [[car alarm]] keeps going off at {L}", "[[motorcycles]] are revving their engines at {L}", "an [[ice cream truck]] idles at {L} with the music on"],
    "noise_commercial": ["the [[nightclub]] at {L} has the [[bass]] up so loud my windows rattle", "the [[restaurant]] at {L} runs a loud [[generator]]"],
    "construction_noise": ["the [[construction site]] at {L} starts [[drilling]] before 7am", "there's [[demolition]] at {L} on a Sunday without a [[permit]]"],
    "heat_hot_water": ["there is [[no heat]] in my apartment at {L}", "we have [[no hot water]] at {L}", "the [[boiler]] at {L} is broken and the [[radiators]] are cold"],
    "unsanitary_condition": ["there's [[mold]] growing on the bathroom wall at {L}", "my kitchen at {L} is full of [[roaches]]", "[[bed bugs]] are in every room at {L}"],
    "rodent": ["[[rats]] are running around the [[playground]] at {L}", "there's a [[rat nest]] by the [[garbage cans]] at {L}"],
    "plumbing_leak": ["[[water]] is pouring through the [[ceiling]] at {L}", "my [[toilet]] at {L} overflows every time", "a [[pipe]] burst in the [[basement]] at {L}"],
    "building_repair": ["the [[front door]] at {L} doesn't lock", "my [[refrigerator]] at {L} stopped working", "the [[stairs]] at {L} are broken", "the [[intercom]] at {L} is dead"],
    "elevator": ["the [[elevator]] at {L} has been out of service", "both [[elevators]] at {L} are broken"],
    "street_condition": ["there's a deep [[pothole]] at {L}", "the [[road]] at {L} has a [[sinkhole]]", "a [[manhole cover]] is missing at {L}"],
    "sidewalk_condition": ["the [[sidewalk]] at {L} is [[cracked]] and raised", "the [[curb]] at {L} is crumbling"],
    "street_light": ["the [[street light]] at {L} is out", "the [[lamp post]] at {L} keeps [[flickering]]"],
    "traffic_signal": ["the [[traffic light]] at {L} is [[dark]]", "the [[walk signal]] at {L} never changes"],
    "water_system": ["a [[hydrant]] is [[open]] at {L}", "there's [[no water]] at {L}", "the [[tap water]] at {L} is [[brown]]"],
    "sewer": ["the [[storm drain]] at {L} is clogged", "[[sewage]] is coming up in the [[basement]] at {L}"],
    "abandoned_vehicle": ["an [[abandoned car]] with [[no plates]] is at {L}", "a [[van]] with [[smashed windows]] has been left at {L}"],
    "dirty_condition": ["somebody [[dumped]] [[tires]] at {L}", "there's [[trash]] piled up on the sidewalk at {L}"],
    "missed_collection": ["[[sanitation]] didn't pick up the [[recycling]] at {L}", "our [[garbage]] wasn't [[collected]] at {L}"],
    "tree": ["a [[branch]] fell on a car at {L}", "a [[dead tree]] at {L} is [[leaning]] over the street"],
    "homeless_assistance": ["a [[man]] is [[sleeping]] in the [[lobby]] at {L} and seems unwell", "there's an [[encampment]] growing at {L}"],
    "illegal_fireworks": ["kids are lighting [[fireworks]] at {L}"],
    "graffiti": ["someone [[tagged]] the [[storefront]] at {L} with [[graffiti]]"],
    "UNKNOWN": ["I have a question about my [[jury duty]] notice", "I want to dispute a [[parking ticket]]", "how do I get a [[marriage certificate]] copy",
                "my [[unemployment]] check never came", "I need help with my [[SNAP benefits]]"],
}

PROF_MOD = ["fucking", "goddamn", "shitty", "damn fucking", "motherfucking", "bloody fucking"]
INSULT_PRE = ["listen here you {n},", "hey {n},", "you stupid {n},", "are you a total {n}?"]
INSULT_NOUN = ["moron", "idiot", "dumbass", "clown", "loser", "imbecile", "dipshit", "numbskull", "halfwit"]
INSULT_SUF = ["you incompetent {n}s", "what a bunch of useless {n}s you are", "you are a joke", "you people are pathetic",
              "you're the worst operator ever", "are you even listening, {n}", "get your head out of your ass"]
SEXUAL = ["hey baby,", "you sound so sexy,", "what are you wearing,", "I bet you look hot,", "talk dirty to me honey,", "come sit on my lap,"]
IDENT = ["put someone on who speaks english,", "go back to where you came from,", "I don't want your kind on the phone,"]
THREAT = ["or I'll come down there and smash your face in", "I'm going to find you after work", "I'll put a bullet in someone",
          "I swear I'll beat the crap out of the super", "you better watch your back", "I'm bringing a bat next time",
          "I'll burn this building down"]
SELF = ["I don't want to live anymore", "I'm thinking of ending my life"]
FIG = ["this is killing me", "I'm dying here", "we're freezing to death", "it's a nightmare", "it's driving me insane",
       "the smell could kill a horse", "I'm going crazy"]
CIVIL_ANGRY = ["I am really upset about this.", "This is unacceptable.", "I'm very frustrated with your agency.", "Nobody ever follows up."]


def facts_of(s):
    return re.findall(r"\[\[(.+?)\]\]", s), re.sub(r"\[\[(.+?)\]\]", r"\1", s)


def make(intent, kind):
    L = R.choice([address(), street(), f"{address()} apartment {apt()}"])
    clause = R.choice(ISSUES[intent]).replace("{A}", apt())
    facts, clause = facts_of(clause)
    loc_fact = L.split(" apartment ")
    clause = clause.replace("{L}", L) if "{L}" in clause else clause
    if "{L}" in R.choice(ISSUES[intent]) or L in clause:
        facts += loc_fact if " apartment " not in L else [loc_fact[0], loc_fact[1]]
    parts, tox = [clause], []
    if R.random() < 0.6:
        t = R.choice(TIMES + DURS); parts.append(t); facts.append(t.replace("since ", "").replace("for ", ""))
    if R.random() < 0.35:
        p = R.choice(PRIOR).format(k=R.choice(KS), o=R.choice(ORDS)); parts.append(p)
        facts.append(re.search(r"(two|three|four|five|six|7|10|second|third|fourth|fifth)", p).group(1))
    if R.random() < 0.3:
        v = R.choice(VULN); parts.append(v); facts.append(" ".join(v.split()[-2:]))
    if R.random() < 0.35:
        q = R.choice(list(REQ)); parts.append(q); facts.append(REQ[q])
    route = "PASS"
    text = ", ".join(parts)
    if kind == "profanity":
        # inline modifier before the first fact noun
        f0 = facts[0]
        i = text.find(f0)
        mod = R.choice(PROF_MOD)
        text = text[:i] + mod + " " + text[i:]
        tox.append(mod); route = "CALM"
    elif kind == "insult":
        if R.random() < 0.5:
            pre = R.choice(INSULT_PRE).format(n=R.choice(INSULT_NOUN)); text = pre + " " + text; tox.append(pre.strip(",?"))
        else:
            suf = R.choice(INSULT_SUF).format(n=R.choice(INSULT_NOUN)); text = text + ", " + suf; tox.append(suf)
        route = "CALM"
    elif kind == "sexual":
        pre = R.choice(SEXUAL); text = pre + " " + text; tox.append(pre.strip(",")); route = "CALM"
    elif kind == "identity":
        pre = R.choice(IDENT); text = pre + " " + text; tox.append(pre.strip(",")); route = "CALM"
    elif kind == "threat":
        th = R.choice(THREAT); text = text + ", " + th; tox.append(th); route = "ESCALATE"
    elif kind == "self_harm":
        th = R.choice(SELF); text = text + ". " + th; route = "ESCALATE"
    elif kind == "figurative":
        text = text + ", " + R.choice(FIG)
    elif kind == "civil_angry":
        text = R.choice(CIVIL_ANGRY) + " " + text
    elif kind == "combo":
        mod = R.choice(PROF_MOD); f0 = facts[0]; i = text.find(f0); text = text[:i] + mod + " " + text[i:]; tox.append(mod)
        suf = R.choice(INSULT_SUF).format(n=R.choice(INSULT_NOUN)); text = text + ", " + suf; tox.append(suf); route = "CALM"
    text = text[0].upper() + text[1:] + "."
    if R.random() < 0.25:  # STT-style
        text = re.sub(r"[,.?!]", "", text.lower())
    return dict(text=text, route=route, route_ok=[route], intent=intent, facts=facts, toxic_spans=tox, source=f"synth:{kind}")


KINDS = [("clean", 0.18), ("civil_angry", 0.07), ("figurative", 0.1), ("profanity", 0.18), ("insult", 0.16), ("sexual", 0.08),
         ("identity", 0.05), ("combo", 0.08), ("threat", 0.08), ("self_harm", 0.02)]
if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    out = sys.argv[3] if len(sys.argv) > 3 else "data/synth/dev_synth.jsonl"
    with open(out, "w") as f:
        for i in range(n):
            kind = R.choices([k for k, _ in KINDS], [w for _, w in KINDS])[0]
            intent = R.choice(list(ISSUES))
            row = make(intent, kind); row["id"] = f"syn-{i:04d}"
            f.write(json.dumps(row) + "\n")
    print("wrote", n, out)
