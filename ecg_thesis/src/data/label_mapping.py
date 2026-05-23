LABEL_MAP = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
}

CLASS_TO_ID = {
    "N": 0,
    "S": 1,
    "V": 2,
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

CLASS_NAMES = ["N", "S", "V"]


def map_symbol(symbol: str) -> int | None:
    mapped = LABEL_MAP.get(symbol)
    if mapped is None:
        return None
    return CLASS_TO_ID[mapped]
