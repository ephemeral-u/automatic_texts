SOURCE_TYPES = {
    "statement": 1,
    "chat_log": 1,
    "regulation": 2
}

STATUS_TYPES = {
    "direct": 0,
    "reported": 1,
    "inferred": 1,
    "disputed": 2
}

BEFORE_EVENT = {
    "C01", "C02", "C03", "C04",
    "C30", "C31", "C32", "C33", "C34", "C37"
}

DURING_EVENT = {
    "C05", "C06", "C07", "C08",
    "C12", "C14", "C15", "C16", "C17",
    "C18", "C19", "C20", "C21",
    "C22", "C23", "C24", "C25",
    "C26", "C27", "C28"
}

EXCEPTION_CLAIMS = {"C31", "C32", "C33", "C34"}


def get_padic_path(claim, document):
    a0 = SOURCE_TYPES.get(document["source_type"], 0)
    a1 = STATUS_TYPES[claim["status"]]

    if claim["claim_id"] in EXCEPTION_CLAIMS:
        a1 = 2

    if claim["claim_id"] in BEFORE_EVENT:
        a2 = 0
    elif claim["claim_id"] in DURING_EVENT:
        a2 = 1
    else:
        a2 = 2

    return (a0, a1, a2)


def padic_code(path):
    a0, a1, a2 = path
    return a0 + a1 * 3 + a2 * 9


def padic_distance(path1, path2):
    if path1 == path2:
        return 0

    k = 0

    for a, b in zip(path1, path2):
        if a != b:
            break
        k += 1

    return 3 ** (-k)


if __name__ == "__main__":
    path1 = (0, 0, 1)
    path2 = (0, 0, 2)
    path3 = (1, 0, 1)

    print("Код первого пути:", padic_code(path1))
    print("Расстояние 1-2:", padic_distance(path1, path2))
    print("Расстояние 1-3:", padic_distance(path1, path3))
    print("Расстояние 1-1:", padic_distance(path1, path1))
