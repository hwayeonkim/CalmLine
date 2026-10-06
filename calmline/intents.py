"""Intent catalog derived from NYC 311 Service Requests (Jun–Aug 2026 volume).

Each intent groups NYC 311 complaint types that a caller would not distinguish
between on the phone. `issue` is the pre-written calm phrase used by the template
renderer; `examples` seed the embedding classifier; `keywords` drive the v0 baseline.
"""

INTENTS = {
    "illegal_parking": dict(
        nyc=["Illegal Parking"],
        issue="a vehicle that is parked illegally",
        keywords=["parked", "parking", "hydrant", "double parked", "bike lane", "crosswalk", "plate"],
        examples=["someone is parked in front of the fire hydrant", "a truck has been double parked blocking traffic all day",
                  "cars keep parking in the bike lane", "there is a car parked on the crosswalk", "commercial truck parked overnight on a residential street"],
    ),
    "blocked_driveway": dict(
        nyc=["Blocked Driveway"],
        issue="a vehicle blocking my driveway",
        keywords=["driveway", "blocking my", "can't get out", "cant get my car out"],
        examples=["somebody parked across my driveway and I can't get my car out", "a van is blocking my driveway", "my driveway is partially blocked again"],
    ),
    "noise_residential": dict(
        nyc=["Noise - Residential", "Noise"],
        issue="a noise problem from a neighboring apartment",
        keywords=["neighbor", "upstairs", "downstairs", "next door", "party", "music", "banging", "stomping", "tv"],
        examples=["my upstairs neighbor plays loud music every night", "the people next door are having a party at 3am",
                  "constant banging and stomping from the apartment above", "neighbor's tv is blasting through the wall"],
    ),
    "noise_street": dict(
        nyc=["Noise - Street/Sidewalk", "Noise - Park"],
        issue="loud noise on the street outside",
        keywords=["outside", "street", "sidewalk", "speaker", "people yelling", "block party", "park"],
        examples=["people are blasting music on the sidewalk outside my window", "a group is yelling on the street all night",
                  "there is a loud party in the park", "someone has a speaker on the corner playing music"],
    ),
    "noise_vehicle": dict(
        nyc=["Noise - Vehicle"],
        issue="noise coming from vehicles",
        keywords=["horn", "honking", "car alarm", "engine", "idling", "motorcycle", "exhaust", "car music"],
        examples=["cars honking nonstop at the intersection", "a car alarm has been going off for an hour",
                  "a truck is idling outside with the engine running", "motorcycles revving their engines all night"],
    ),
    "noise_commercial": dict(
        nyc=["Noise - Commercial"],
        issue="noise from a nearby business",
        keywords=["bar", "club", "restaurant", "store", "business", "lounge", "bodega"],
        examples=["the bar downstairs plays music until 4am", "the club on the corner is way too loud",
                  "the restaurant's exhaust fan is extremely loud"],
    ),
    "construction_noise": dict(
        nyc=["Noise - Construction"],
        issue="construction work happening outside permitted hours",
        keywords=["construction", "jackhammer", "drilling", "crane", "after hours", "permit"],
        examples=["construction crew is jackhammering at 6am on a sunday", "they are drilling late at night next door",
                  "construction is happening without a permit"],
    ),
    "heat_hot_water": dict(
        nyc=["HEAT/HOT WATER"],
        issue="no heat or hot water in my building",
        keywords=["heat", "hot water", "heating", "radiator", "boiler", "freezing", "cold water"],
        examples=["we have had no heat in the building for three days", "there is no hot water in my apartment",
                  "the radiator doesn't work and it's freezing", "landlord won't fix the boiler"],
    ),
    "unsanitary_condition": dict(
        nyc=["UNSANITARY CONDITION"],
        issue="unsanitary conditions in my apartment, such as pests or mold",
        keywords=["mold", "roaches", "cockroaches", "bed bugs", "pests", "mice", "mildew", "smell"],
        examples=["there is black mold all over the bathroom", "the apartment is infested with roaches",
                  "bed bugs in my apartment and the landlord does nothing", "mice everywhere in the kitchen"],
    ),
    "rodent": dict(
        nyc=["Rodent"],
        issue="rats or rodents in the area",
        keywords=["rats", "rat", "rodent", "rodents"],
        examples=["there are rats all over the sidewalk", "huge rats coming out of the garbage on my block", "rats in the backyard"],
    ),
    "plumbing_leak": dict(
        nyc=["PLUMBING", "WATER LEAK", "General Construction/Plumbing"],
        issue="a plumbing problem or water leak in my building",
        keywords=["leak", "leaking", "pipe", "toilet", "sink", "ceiling", "flooding", "drip", "shower"],
        examples=["water is leaking through my ceiling from upstairs", "the toilet has been broken for a week",
                  "a pipe burst in the bathroom", "the sink is clogged and the landlord won't fix it"],
    ),
    "building_repair": dict(
        nyc=["PAINT/PLASTER", "DOOR/WINDOW", "FLOORING/STAIRS", "APPLIANCE", "ELECTRIC", "GENERAL"],
        issue="a repair problem inside my apartment that the landlord has not fixed",
        keywords=["landlord", "repair", "window", "door", "floor", "stove", "fridge", "refrigerator", "outlet",
                  "intercom", "buzzer", "plaster", "paint", "lock", "power"],
        examples=["the landlord won't fix my broken window", "the ceiling plaster is falling down", "my stove has not worked for a month",
                  "the front door lock is broken", "the intercom doesn't work", "the outlets in my apartment have no power"],
    ),
    "elevator": dict(
        nyc=["Elevator"],
        issue="a broken elevator in my building",
        keywords=["elevator", "lift"],
        examples=["the elevator in my building has been broken for two weeks", "elevator is out and I live on the 12th floor"],
    ),
    "street_condition": dict(
        nyc=["Street Condition"],
        issue="a damaged road surface, such as a pothole",
        keywords=["pothole", "road", "cave-in", "sinkhole", "asphalt", "manhole", "street is broken"],
        examples=["there is a huge pothole on my street", "the road has caved in near the intersection",
                  "a pothole damaged my tire", "a manhole cover is loose and rattling"],
    ),
    "sidewalk_condition": dict(
        nyc=["Sidewalk Condition", "Curb Condition", "Root/Sewer/Sidewalk Condition"],
        issue="a broken sidewalk or curb",
        keywords=["sidewalk", "curb", "tripped", "cracked"],
        examples=["the sidewalk is cracked and someone tripped", "broken curb in front of my building", "tree roots lifted the sidewalk"],
    ),
    "street_light": dict(
        nyc=["Street Light Condition"],
        issue="a street light that is out or damaged",
        keywords=["street light", "streetlight", "lamp post", "light is out", "dark"],
        examples=["the street light on my corner has been out for weeks", "the whole block is dark because the lights are out"],
    ),
    "traffic_signal": dict(
        nyc=["Traffic Signal Condition"],
        issue="a traffic signal that is not working properly",
        keywords=["traffic light", "signal", "walk sign", "red light", "stop light"],
        examples=["the traffic light at the intersection is not working", "the walk signal is stuck", "traffic signal is flashing red"],
    ),
    "water_system": dict(
        nyc=["Water System", "Water Maintenance"],
        issue="a problem with the water supply or a fire hydrant",
        keywords=["hydrant", "no water", "water main", "water pressure", "water is brown", "open hydrant"],
        examples=["a fire hydrant is open and flooding the street", "we have no water in the whole building",
                  "the tap water is brown", "a water main broke on our street"],
    ),
    "sewer": dict(
        nyc=["Sewer", "Sewer Maintenance"],
        issue="a sewer backup or clogged catch basin",
        keywords=["sewer", "catch basin", "drain", "storm drain", "sewage", "backup"],
        examples=["the catch basin is clogged and the street is flooding", "sewage is backing up into the basement", "storm drain is blocked"],
    ),
    "abandoned_vehicle": dict(
        nyc=["Abandoned Vehicle", "Derelict Vehicles"],
        issue="an abandoned vehicle on the street",
        keywords=["abandoned", "hasn't moved", "derelict", "flat tires", "no plates", "car has been sitting"],
        examples=["a car has been abandoned on my block for months", "there's a car with no plates and flat tires", "a derelict van hasn't moved in weeks"],
    ),
    "dirty_condition": dict(
        nyc=["Dirty Condition", "Illegal Dumping", "Street Sweeping Complaint"],
        issue="litter or illegally dumped trash",
        keywords=["dumping", "dumped", "litter", "trash", "garbage bags", "filthy", "mattress", "debris"],
        examples=["someone dumped a mattress and furniture on the sidewalk", "the street is covered with trash", "people keep dumping garbage in the empty lot"],
    ),
    "missed_collection": dict(
        nyc=["Missed Collection"],
        issue="a missed trash, recycling, or compost pickup",
        keywords=["pickup", "pick up", "collection", "sanitation", "recycling", "compost", "bulk"],
        examples=["sanitation skipped our garbage pickup again", "recycling hasn't been picked up in two weeks", "they missed the bulk item pickup"],
    ),
    "tree": dict(
        nyc=["Damaged Tree", "Overgrown Tree/Branches", "Dead/Dying Tree"],
        issue="a damaged, fallen, or overgrown tree",
        keywords=["tree", "branch", "branches", "limb", "fallen tree", "uprooted"],
        examples=["a big branch fell and is blocking the sidewalk", "a dead tree is about to fall on my house", "tree branches are hitting the power lines"],
    ),
    "homeless_assistance": dict(
        nyc=["Encampment", "Homeless Person Assistance"],
        issue="a person experiencing homelessness who may need assistance",
        keywords=["homeless", "encampment", "sleeping on", "tent", "unhoused"],
        examples=["there is a homeless encampment under the bridge", "a man has been sleeping in my building's doorway and needs help",
                  "a homeless person looks sick on the subway entrance"],
    ),
    "illegal_fireworks": dict(
        nyc=["Illegal Fireworks"],
        issue="illegal fireworks being set off",
        keywords=["fireworks", "firecrackers", "explosions"],
        examples=["people are setting off fireworks every night", "fireworks going off in the street at 2am"],
    ),
    "graffiti": dict(
        nyc=["Graffiti"],
        issue="graffiti that needs to be removed",
        keywords=["graffiti", "tagged", "spray paint", "vandalized"],
        examples=["someone spray painted graffiti all over our building", "the wall was tagged with graffiti again"],
    ),
}

UNKNOWN = "UNKNOWN"
INTENT_KEYS = list(INTENTS)


def nyc_coverage(type_counts: dict) -> float:
    covered = {t for v in INTENTS.values() for t in v["nyc"]}
    tot = sum(type_counts.values())
    return sum(n for t, n in type_counts.items() if t in covered) / max(tot, 1)
