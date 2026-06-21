"""
labels.py - maps class folder names (as used by torchvision.ImageFolder) to
human-readable flower names, and provides fun facts for the frontend.

FlowerLens AI is trained on the Oxford 102 Flowers dataset, where the train/
and valid/ folders are named numerically ("1".."102"). The real number->name
mapping for this exact dataset ships with it as cat_to_name.json. We bake the
same mapping in here so the backend never depends on that file being present
at inference time (e.g. on Render, where only the backend/ folder is deployed).

If a different dataset is ever swapped in whose folders are already named with
real flower words ("rose", "sunflower", ...), get_name() detects that
automatically and just title-cases the folder name instead of consulting the
dict below. This keeps labels.py dataset-agnostic.
"""

import os
import json

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLASS_NAMES_PATH = os.path.join(_THIS_DIR, "class_names.json")

# Real Oxford-102 folder-number -> flower-name mapping, copied from the
# cat_to_name.json that ships with this dataset. Used whenever folder names
# are plain numbers (which is the case for this project's dataset).
FLOWER_LABELS = {
    "1": "pink primrose", "2": "hard-leaved pocket orchid", "3": "canterbury bells",
    "4": "sweet pea", "5": "english marigold", "6": "tiger lily", "7": "moon orchid",
    "8": "bird of paradise", "9": "monkshood", "10": "globe thistle",
    "11": "snapdragon", "12": "colt's foot", "13": "king protea", "14": "spear thistle",
    "15": "yellow iris", "16": "globe-flower", "17": "purple coneflower",
    "18": "peruvian lily", "19": "balloon flower", "20": "giant white arum lily",
    "21": "fire lily", "22": "pincushion flower", "23": "fritillary",
    "24": "red ginger", "25": "grape hyacinth", "26": "corn poppy",
    "27": "prince of wales feathers", "28": "stemless gentian", "29": "artichoke",
    "30": "sweet william", "31": "carnation", "32": "garden phlox",
    "33": "love in the mist", "34": "mexican aster", "35": "alpine sea holly",
    "36": "ruby-lipped cattleya", "37": "cape flower", "38": "great masterwort",
    "39": "siam tulip", "40": "lenten rose", "41": "barbeton daisy",
    "42": "daffodil", "43": "sword lily", "44": "poinsettia", "45": "bolero deep blue",
    "46": "wallflower", "47": "marigold", "48": "buttercup", "49": "oxeye daisy",
    "50": "common dandelion", "51": "petunia", "52": "wild pansy", "53": "primula",
    "54": "sunflower", "55": "pelargonium", "56": "bishop of llandaff",
    "57": "gaura", "58": "geranium", "59": "orange dahlia", "60": "pink-yellow dahlia",
    "61": "cautleya spicata", "62": "japanese anemone", "63": "black-eyed susan",
    "64": "silverbush", "65": "californian poppy", "66": "osteospermum",
    "67": "spring crocus", "68": "bearded iris", "69": "windflower",
    "70": "tree poppy", "71": "gazania", "72": "azalea", "73": "water lily",
    "74": "rose", "75": "thorn apple", "76": "morning glory", "77": "passion flower",
    "78": "lotus", "79": "toad lily", "80": "anthurium", "81": "frangipani",
    "82": "clematis", "83": "hibiscus", "84": "columbine", "85": "desert-rose",
    "86": "tree mallow", "87": "magnolia", "88": "cyclamen", "89": "watercress",
    "90": "canna lily", "91": "hippeastrum", "92": "bee balm", "93": "ball moss",
    "94": "foxglove", "95": "bougainvillea", "96": "camellia", "97": "mallow",
    "98": "mexican petunia", "99": "bromelia", "100": "blanket flower",
    "101": "trumpet creeper", "102": "blackberry lily",
}

FLOWER_FACTS = {
    "rose": "Roses have been cultivated for over 5,000 years and symbolize love in over 50 cultures.",
    "sunflower": "Sunflowers track the sun across the sky - a behaviour called heliotropism.",
    "lotus": "The Lotus closes at night and sinks underwater, rising again each morning.",
    "daisy": "The word daisy comes from Old English 'daes eage' meaning 'day's eye'.",
    "dandelion": "Every part of a dandelion is edible - roots, leaves, and flowers.",
    "tulip": "In 1637 tulip bulbs in Holland were worth more than gold - called Tulip Mania.",
    "orchid": "Orchids are the largest family of flowering plants with over 25,000 species.",
    "iris": "The iris was named after the Greek goddess of the rainbow due to its many colours.",
}

_DEFAULT_FACT = "Flowers have been used by humans for food, medicine, and decoration for thousands of years."


def _looks_like_number(name: str) -> bool:
    return str(name).strip().isdigit()


def _load_class_names():
    """Read class_names.json (written by 3_train.py) if it exists yet."""
    if os.path.exists(_CLASS_NAMES_PATH):
        try:
            with open(_CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


_CLASS_NAMES = _load_class_names()


def get_name(class_id_or_folder_name) -> str:
    """
    Always returns a human-readable flower name.

    Accepts either:
      - an int index into class_names.json (e.g. 53), as produced by the model
      - a raw folder name string (e.g. "74" or "rose")
    """
    if isinstance(class_id_or_folder_name, int):
        if 0 <= class_id_or_folder_name < len(_CLASS_NAMES):
            folder_name = str(_CLASS_NAMES[class_id_or_folder_name])
        else:
            folder_name = str(class_id_or_folder_name)
    else:
        folder_name = str(class_id_or_folder_name)

    if _looks_like_number(folder_name):
        mapped = FLOWER_LABELS.get(folder_name)
        if mapped:
            return mapped.title()
        return f"Class {folder_name}"

    # Folder name is already a readable flower word/phrase
    return folder_name.replace("_", " ").title()


def get_fact(class_name: str) -> str:
    """Returns a fun fact for a (display) flower name, case-insensitive lookup."""
    key = str(class_name).strip().lower()
    if key in FLOWER_FACTS:
        return FLOWER_FACTS[key]
    # fall back to substring match, e.g. "Common Dandelion" -> contains "dandelion"
    for fact_key, fact in FLOWER_FACTS.items():
        if fact_key in key:
            return fact
    return _DEFAULT_FACT
