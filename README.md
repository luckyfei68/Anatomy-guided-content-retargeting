# Anatomy-guided content retargeting for chest CT preview

Materials accompanying the manuscript *Anatomy-guided content retargeting for chest CT preview*.

## Files

- `results/`: five numerical manuscript tables in CSV format.
- `code/seam_carving.py`: the project's core Sobel energy, seam search, and seam removal functions.
- `code/config.py`: the protection weight used in the manuscript.
- `code/demo.py`: a small synthetic-image example showing how to call the functions.
- `SHA256SUMS.txt`: file checksums.

The example uses a generated array; it does not use patient data. The CSV tables preserve the manuscript's displayed rounding.

## Quick example

Install NumPy and SciPy, then run `python code/demo.py` from the repository root.
