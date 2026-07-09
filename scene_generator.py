"""
Level 3: turn detected sound label(s) into a natural-language scene
description.

Approach: rule-based, not learned. Two layers:
  1. A curated CURATED_SCENARIOS dict for specific, evocative combinations
     worth writing by hand (e.g. rain + thunderstorm).
  2. A CATEGORY_TEMPLATES fallback keyed by which of ESC-50's 5 broad
     categories each label belongs to, so ANY combination of two detected
     labels gets a coherent sentence, not just the ones curated by hand --
     this makes the system robust to all ~1225 possible label pairs without
     manually writing a template for every one.

This is deterministic and demo-reliable, which is what was recommended
earlier over calling an LLM for this step (no added dependency, predictable
output, no risk of hallucinated scenes).

Usage:
    from scene_generator import generate_scenario
    generate_scenario(["dog"])                      -> single-label sentence
    generate_scenario(["dog", "rain"])               -> two-label sentence
    generate_scenario(["children_playing", "rain"])  -> curated, if present

Phrasing variety:
    VARY_PHRASING toggles between deterministic output (same pair always
    produces the same sentence -- good for reproducible demos/tests) and
    randomized output (each fallback template has 2-3 phrasings, one
    picked at random per call -- good for a livelier final demo). Curated
    scenarios are unaffected either way; they're hand-written one-offs, not
    templates. Pass generate_scenario(labels, seed=42) for a reproducible
    "random" pick if you want variety but still need repeatable tests.
"""
import random

VARY_PHRASING = True

# ESC-50's 5 official broad categories, used for the fallback templates.
CATEGORY_MAP = {
    # Animals
    "dog": "animal", "rooster": "animal", "pig": "animal", "cow": "animal",
    "frog": "animal", "cat": "animal", "hen": "animal", "insects": "animal",
    "sheep": "animal", "crow": "animal",

    # Natural soundscapes & water sounds
    "rain": "nature", "sea_waves": "nature", "crackling_fire": "nature",
    "crickets": "nature", "chirping_birds": "nature", "water_drops": "nature",
    "wind": "nature", "pouring_water": "nature", "toilet_flush": "nature",
    "thunderstorm": "nature",

    # Human, non-speech sounds
    "crying_baby": "human", "sneezing": "human", "clapping": "human",
    "breathing": "human", "coughing": "human", "footsteps": "human",
    "laughing": "human", "brushing_teeth": "human", "snoring": "human",
    "drinking_sipping": "human",

    # Interior/domestic sounds
    "door_wood_knock": "domestic", "mouse_click": "domestic",
    "keyboard_typing": "domestic", "door_wood_creaks": "domestic",
    "can_opening": "domestic", "washing_machine": "domestic",
    "vacuum_cleaner": "domestic", "clock_alarm": "domestic",
    "clock_tick": "domestic", "glass_breaking": "domestic",

    # Exterior/urban noises
    "helicopter": "urban", "chainsaw": "urban", "siren": "urban",
    "car_horn": "urban", "engine": "urban", "train": "urban",
    "church_bells": "urban", "airplane": "urban", "fireworks": "urban",
    "hand_saw": "urban",
}

# Hand-picked, specific combinations worth writing a dedicated sentence for.
# Key = frozenset of the two class names (order doesn't matter).
# Add more here as you find combinations worth calling out specifically.
CURATED_SCENARIOS = {
    frozenset({"rain", "thunderstorm"}): "A storm is rolling in, with rain falling as thunder rumbles.",
    frozenset({"crying_baby", "rain"}): "A baby is crying while rain falls outside.",
    frozenset({"footsteps", "rain"}): "Someone is walking through the rain.",
    frozenset({"dog", "rain"}): "A dog is barking outside in the rain.",
    frozenset({"laughing", "clapping"}): "People are laughing and clapping together.",
    frozenset({"footsteps", "laughing"}): "Someone is walking while laughing.",
    frozenset({"fireworks", "church_bells"}): "Fireworks are going off as church bells ring.",
    frozenset({"chainsaw", "chirping_birds"}): "A chainsaw is cutting through the woods as birds chirp nearby.",
    frozenset({"engine", "car_horn"}): "A car engine is running as a horn honks.",
    frozenset({"siren", "engine"}): "An emergency vehicle's siren blares over the engine noise.",
    frozenset({"washing_machine", "footsteps"}): "A washing machine runs in the background as someone walks by.",
    frozenset({"keyboard_typing", "mouse_click"}): "Someone is typing and clicking at a computer.",
    frozenset({"coughing", "sneezing"}): "Someone is coughing and sneezing.",
    frozenset({"crackling_fire", "wind"}): "A fire crackles as wind blows outside.",
    frozenset({"sea_waves", "chirping_birds"}): "Waves roll onto the shore as birds chirp nearby.",

    # Added for broader coverage / more vivid scenes.
    frozenset({"rooster", "hen"}): "A rooster crows on the farm as a hen clucks nearby.",
    frozenset({"cow", "sheep"}): "Cows and sheep can be heard grazing on the farm.",
    frozenset({"dog", "footsteps"}): "A dog barks as someone approaches on foot.",
    frozenset({"cat", "laughing"}): "A cat meows while someone laughs nearby.",
    frozenset({"crackling_fire", "laughing"}): "People are laughing around a crackling campfire.",
    frozenset({"insects", "crickets"}): "The night is alive with the sound of insects and crickets.",
    frozenset({"chirping_birds", "wind"}): "Birds chirp as the wind rustles through the trees.",
    frozenset({"frog", "insects"}): "Frogs and insects fill the night air by the pond.",
    frozenset({"dog", "thunderstorm"}): "A dog barks nervously as thunder rolls in the distance.",
    frozenset({"vacuum_cleaner", "dog"}): "A dog barks at the vacuum cleaner running indoors.",
    frozenset({"siren", "car_horn"}): "A siren wails over the honking of car horns in traffic.",
    frozenset({"helicopter", "siren"}): "A helicopter circles overhead as sirens sound below.",
    frozenset({"engine", "train"}): "A train engine rumbles past.",
    frozenset({"footsteps", "door_wood_knock"}): "Footsteps approach, followed by a knock on the door.",
    frozenset({"crying_baby", "footsteps"}): "A baby cries as someone paces nearby.",
    frozenset({"clapping", "church_bells"}): "Church bells ring as a crowd claps outside.",
    frozenset({"keyboard_typing", "clock_tick"}): "Someone types at a keyboard as a clock ticks in the quiet office.",
    frozenset({"pouring_water", "glass_breaking"}): "Water is being poured just before a glass shatters.",
    frozenset({"snoring", "clock_alarm"}): "An alarm clock rings, interrupting someone's snoring.",
    frozenset({"brushing_teeth", "water_drops"}): "Someone is brushing their teeth as water drips nearby.",
}

# Fallback sentence templates, keyed by the (unordered) pair of broad
# categories. {a} and {b} are filled with the two detected class names
# (formatted: underscores -> spaces, with articles applied where needed).
# Every template wraps names in a "the sound(s) of ..." / "can be heard"
# construction rather than gluing "sounds" directly onto {a}/{b} -- that
# keeps grammar correct regardless of whether a name carries an article
# ("a dog") or is bare ("rain", "footsteps", "fireworks").
CATEGORY_TEMPLATES = {
    frozenset({"animal", "animal"}): [
        "The sounds of {a} and {b} can be heard together.",
        "{a} and {b} can be heard together.",
    ],
    frozenset({"animal", "nature"}): [
        "The sound of {a} mixes with {b} in the background.",
        "{a} can be heard against a backdrop of {b}.",
    ],
    frozenset({"animal", "human"}): [
        "{a} and {b} can both be heard nearby.",
        "{a} can be heard alongside {b}.",
    ],
    frozenset({"animal", "domestic"}): [
        "The sound of {a} can be heard indoors, along with {b}.",
        "Indoors, {a} can be heard along with {b}.",
    ],
    frozenset({"animal", "urban"}): [
        "The sound of {a} cuts through the noise of {b}.",
        "{a} can be heard over the noise of {b}.",
    ],

    frozenset({"nature", "nature"}): [
        "The sounds of {a} and {b} fill the outdoor scene.",
        "{a} and {b} can both be heard outdoors.",
    ],
    frozenset({"nature", "human"}): [
        "{a} and {b} can both be heard outside.",
        "Outside, {a} and {b} can both be heard.",
    ],
    frozenset({"nature", "domestic"}): [
        "{a} can be heard outside while {b} can be heard indoors.",
        "{b} can be heard indoors as {a} continues outside.",
    ],
    frozenset({"nature", "urban"}): [
        "{a} blends with the sound of {b} outdoors.",
        "{a} mixes with {b} outdoors.",
    ],

    frozenset({"human", "human"}): [
        "{a} and {b} can both be heard nearby.",
        "{a} can be heard along with {b}.",
    ],
    frozenset({"human", "domestic"}): [
        "{a} can be heard indoors, along with {b}.",
        "Indoors, {a} can be heard along with {b}.",
    ],
    frozenset({"human", "urban"}): [
        "{a} can be heard while {b} is audible in the background.",
        "{a} can be heard as {b} continues in the background.",
    ],

    frozenset({"domestic", "domestic"}): [
        "Indoors, {a} and {b} can both be heard.",
        "{a} and {b} can both be heard indoors.",
    ],
    frozenset({"domestic", "urban"}): [
        "{a} can be heard indoors while {b} can be heard outside.",
        "{b} can be heard outside as {a} continues indoors.",
    ],

    frozenset({"urban", "urban"}): [
        "The sounds of {a} and {b} fill the urban scene.",
        "{a} and {b} can both be heard in the urban scene.",
    ],
}

SINGLE_TEMPLATES = {
    "animal": [
        "The sound of {a} is heard.",
        "{a} can be heard nearby.",
    ],
    "nature": [
        "{a} can be heard in the environment.",
        "The sound of {a} fills the air.",
    ],
    "human": [
        "The sound of {a} is heard.",
        "{a} can be heard nearby.",
    ],
    "domestic": [
        "The sound of {a} is heard indoors.",
        "{a} can be heard indoors.",
    ],
    "urban": [
        "{a} can be heard nearby.",
        "The sound of {a} carries through the air.",
    ],
}


# A few class names read awkwardly when just underscores->spaces (e.g.
# "drinking sipping" repeats the same idea twice). Special-case those.
NAME_OVERRIDES = {
    "drinking_sipping": "sipping a drink",
    "door_wood_knock": "a knock on a wooden door",
    "door_wood_creaks": "a creaking wooden door",
    "clock_tick": "a ticking clock",
    "clock_alarm": "an alarm clock",
    "car_horn": "a car horn",
    "hand_saw": "a hand saw",
    "washing_machine": "a washing machine",
    "vacuum_cleaner": "a vacuum cleaner",
    "crying_baby": "a crying baby",
    "chirping_birds": "birds chirping",
    "crackling_fire": "a crackling fire",
    "sea_waves": "sea waves",
    "pouring_water": "water being poured",
    "toilet_flush": "a toilet flushing",
    "brushing_teeth": "someone brushing their teeth",
    "keyboard_typing": "keyboard typing",
    "mouse_click": "a mouse click",
    "can_opening": "a can opening",
    "glass_breaking": "glass breaking",
}

# Countable-singular sounds that read naturally with "a"/"an" in front when
# used as a bare subject (dog, cat, engine...). Gerunds/mass nouns/plurals
# (rain, wind, clapping, footsteps, church_bells, fireworks...) don't need
# one and are left alone.
ARTICLE_A = {
    "dog", "cat", "hen", "pig", "cow", "frog", "sheep", "crow", "rooster",
    "helicopter", "chainsaw", "siren", "train", "thunderstorm",
}
ARTICLE_AN = {"airplane", "engine"}


def _format_name(class_name):
    if class_name in NAME_OVERRIDES:
        return NAME_OVERRIDES[class_name]
    name = class_name.replace("_", " ")
    if class_name in ARTICLE_A:
        return f"a {name}"
    if class_name in ARTICLE_AN:
        return f"an {name}"
    return name


def _pick_template(templates, rng):
    """templates: a single string OR a list of phrasing strings. Picks one
    according to VARY_PHRASING -- first entry if variety is off (keeps
    behavior deterministic), otherwise a random entry from the given rng."""
    if isinstance(templates, str):
        return templates
    if not VARY_PHRASING:
        return templates[0]
    return rng.choice(templates)


def generate_scenario(labels, seed=None):
    """
    labels: list/set of ESC-50 class name strings (e.g. ["dog", "rain"]).
    seed: optional int. Only matters when VARY_PHRASING=True -- pass the
          same seed to get the same "random" phrasing back reproducibly
          (e.g. for tests). Omit for a fresh random pick each call.
    Returns a natural-language scene description string.
    """
    rng = random.Random(seed) if seed is not None else random
    labels = list(dict.fromkeys(labels))  # dedupe, preserve order

    if len(labels) == 0:
        return "No sound detected."

    if len(labels) == 1:
        cls = labels[0]
        category = CATEGORY_MAP.get(cls, "urban")
        template = _pick_template(SINGLE_TEMPLATES[category], rng)
        sentence = template.format(a=_format_name(cls))
        return sentence[0].upper() + sentence[1:]

    # Exactly 2 (or more, but we only ever detect up to 2 -- take the first two)
    cls_a, cls_b = labels[0], labels[1]

    curated = CURATED_SCENARIOS.get(frozenset({cls_a, cls_b}))
    if curated:
        return curated

    cat_a = CATEGORY_MAP.get(cls_a, "urban")
    cat_b = CATEGORY_MAP.get(cls_b, "urban")
    templates = CATEGORY_TEMPLATES.get(frozenset({cat_a, cat_b}))
    if not templates:
        # Shouldn't happen given the full 5x5 coverage above, but a safe
        # generic fallback in case a class name isn't in CATEGORY_MAP at all.
        sentence = f"{_format_name(cls_a)} and {_format_name(cls_b)} are both audible in this scene."
    else:
        template = _pick_template(templates, rng)
        sentence = template.format(a=_format_name(cls_a), b=_format_name(cls_b))

    return sentence[0].upper() + sentence[1:]


if __name__ == "__main__":
    # Quick manual sanity check -- includes the two previously-buggy cases
    # (church_bells, thunderstorm) and an animal-first vs animal-second
    # ordering check for the fixed category templates.
    tests = [
        ["dog"],
        ["thunderstorm"],
        ["church_bells"],
        ["dog", "rain"],
        ["rain", "dog"],          # order shouldn't matter
        ["laughing", "clapping"],
        ["chainsaw", "wind"],
        ["mouse_click", "keyboard_typing"],
        ["helicopter", "sheep"],
        ["dog", "cat"],           # animal-animal, previously broken
        ["dog", "washing_machine"],  # animal-domestic, previously broken
        ["dog", "car_horn"],      # animal-urban, previously broken
    ]
    for t in tests:
        print(f"{t}  ->  {generate_scenario(t)}")

    print(f"\nVARY_PHRASING = {VARY_PHRASING} -- calling the same pair 4x:")
    for _ in range(4):
        print(f"  {generate_scenario(['dog', 'car_horn'])}")

    print("\nSame call with a fixed seed=1 -- should repeat identically:")
    for _ in range(3):
        print(f"  {generate_scenario(['dog', 'car_horn'], seed=1)}")