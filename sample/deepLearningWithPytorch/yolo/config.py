DATA_WIDTH = 416
DATA_HEIGHT = 416
CLASS_NUM = 3
ANCHORS = {
    13: [[168, 302], [57, 221], [336, 284]],
    26: [[175, 225], [279, 160], [249, 271]],
    52: [[129, 209], [85, 413], [44, 42]]
}

ANCHORS_AREA = {
    13: [x*y for x, y in ANCHORS[13]],
    26: [x*y for x, y in ANCHORS[26]],
    52: [x*y for x, y in ANCHORS[52]]
}


if __name__ == "__main__":
    print(ANCHORS_AREA[13])