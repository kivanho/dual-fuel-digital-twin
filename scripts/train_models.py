"""
Compatibility wrapper: training now lives in `app.train`.

    python -m scripts.train_models [--out models/synthetic] [--csv map.csv] [--report-dir docs] ...

See `python -m app.train --help`.
"""
import sys

from app.train import main

if __name__ == "__main__":
    sys.exit(main())
