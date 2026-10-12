"""Illustrative run on a synthetic array; no CT images are distributed."""

import numpy as np

from agcr_core import retarget


def main():
    height, width, seams = 32, 48, 8
    image = np.tile(np.linspace(-1000, 100, width, dtype=np.float32), (height, 1))
    lung = np.zeros((height, width), dtype=bool)
    lung[6:26, 12:37] = True

    for name, weight in (("Basic SC", 0.0), ("AGCR", 10.0)):
        output, paths = retarget(image, lung, seams=seams, weight=weight)
        assert output.shape == (height, width - seams)
        assert paths.shape == (seams, height)
        print(name, output.shape, "saved seams:", len(paths))


if __name__ == "__main__":
    main()
