# SPDX-FileCopyrightText: SAS research group, HFT, Helmut Schmidt University
# SPDX-License-Identifier: CC0-1.0
# https://github.com/hsu-sonar/icua24-geopackage

import sys
from pathlib import Path

import numpy as np
from fudgeo.geopkg import GeoPackage

# Library path, relative to the directory containing this file.
libdir = Path(__file__).parent.parent.resolve() / "library"

sys.path.insert(0, str(libdir))

# Import the library function to measure the runtime of find_cd_partners.
from change_detection import measure_runtime

# Benchmark databases are in the same directory as this file.
benchmark_dir = Path(__file__).parent.resolve()

# Input databases and results filenames.
db50_geo = benchmark_dir / "benchmark_missions_50_geographic.gpkg"
results_fn_geo50 = benchmark_dir / "change_detection_50_geographic.npy"
db50_cart = benchmark_dir / "benchmark_missions_50_cartesian.gpkg"
results_fn_cart50 = benchmark_dir / "change_detection_50_cartesian.npy"
db500_geo = benchmark_dir / "benchmark_missions_500_geographic.gpkg"
results_fn_geo500 = benchmark_dir / "change_detection_500_geographic.npy"
db500_cart = benchmark_dir / "benchmark_missions_500_cartesian.gpkg"
results_fn_cart500 = benchmark_dir / "change_detection_500_cartesian.npy"

all_db = [
    (db50_geo, results_fn_geo50),
    (db50_cart, results_fn_cart50),
    (db500_geo, results_fn_geo500),
    (db500_cart, results_fn_cart500),
]

# Load each, load the spatialite extension, and benchmark them.
for db, fn in all_db:
    print("Benchmarking", db)
    gpkg = GeoPackage(db)
    gpkg.connection.enable_load_extension(True)
    gpkg.connection.execute("SELECT load_extension('mod_spatialite');")

    results = measure_runtime(gpkg, 10000)
    np.save(fn, results)
