

ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

ACTION_MEANING_SHORT = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
}


def perms(l):
    if l == 1:
        return [(False,), (True,)]
    else:
        p = perms(l - 1)
        out = []
        for x in p:
            out.append((*x, False))
            out.append((*x, True))
        return out


def create_map(short=False):
    am = ACTION_MEANING_SHORT if short else ACTION_MEANING
    inv_map = {v: k for k, v in am.items()}
    out = {}
    for x in perms(5):
        key = ""
        if not short:
            if x[1] ^ x[4]:
                key += "UP" if x[1] else "DOWN"
        if x[2] ^ x[3]:
            key += "RIGHT" if x[2] else "LEFT"
        if x[0]:
            key += "FIRE"
        if not key:
            key = "NOOP"
        out[x] = inv_map[key]
    return out


# maps the boolean vector '(FIRE, UP, RIGHT,LEFT, DOWN)' to the discrete action space of the atari
CONTROLLER_TO_ACTION = create_map()
CONTROLLER_TO_ACTION_SHORT = create_map(True)
