"""Constants shared by training, evaluation, the solver and the demo app.

Label convention (must match how pairs.csv was generated):
    for a pair (A, B) the label says where tile B sits relative to tile A.
        LEFT  -> B is directly to the left of A
        RIGHT -> B is directly to the right of A
        UP    -> B is directly above A
        DOWN  -> B is directly below A
"""

NO_MATCH_IDX, LEFT_IDX, RIGHT_IDX, UP_IDX, DOWN_IDX = 0, 1, 2, 3, 4
NUM_CLASSES = 5

LABEL_NAMES = {0: "NO_MATCH", 1: "LEFT", 2: "RIGHT", 3: "UP", 4: "DOWN"}
CLASS_LABEL = {"no_match": 0, "LEFT": 1, "RIGHT": 2, "UP": 3, "DOWN": 4}

# class id -> (d_row, d_col) of tile B relative to tile A
DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}

TILE_SIZE = 128

# Channel statistics of the training pairs (the values used to train the released model).
MEAN = [0.5452349781990051, 0.6246451139450073, 0.4715678095817566]
STD = [0.23454082012176514, 0.19042210280895233, 0.21782392263412476]
