import numpy as np

# np.set_printoptions(threshold=np.inf)

BLOCK_SIZE = 8
LOW_FREQUENCY_THRESHOLD = 8
ELEMENTS_PER_BIN = 42000

def _build_frequency_map():
    m = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            m[i, j] = i + j != 0 and i + j < LOW_FREQUENCY_THRESHOLD
    return m


LOW_FREQUENCY_MAP = _build_frequency_map()
