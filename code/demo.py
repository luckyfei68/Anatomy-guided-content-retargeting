"""Illustrative run on a synthetic array; no CT images are distributed."""

import numpy as np

from seam_carving import energy_basic, energy_medical, run_seam_carving


def main():
    height, width, seams = 32, 48, 8
    image = np.tile(np.linspace(-1000, 100, width, dtype=np.float32), (height, 1))
    lung = np.zeros((height, width), dtype=bool)
    lung[6:26, 12:37] = True

    for name, energy in (("Basic SC", energy_basic), ("AGCR", energy_medical)):
        output, paths = run_seam_carving(
            image, energy, seams, lung_mask=lung, verbose=False
        )
        assert output.shape == (height, width - seams)
        assert len(paths) == seams and all(len(path) == height for path in paths)
        print(name, output.shape, "saved seams:", len(paths))


if __name__ == "__main__":
    main()
