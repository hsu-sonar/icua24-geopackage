<!--

SPDX-FileCopyrightText: SAS research group, HFT, Helmut Schmidt University
SPDX-License-Identifier: CC0-1.0

-->

Benchmarking scripts
====================

The scripts in this file run timing benchmarks on the application algorithms in the
library. They use the same databases as the benchmarks reported in the paper; these
databases are in this directory. The database filenames give the number of mission
groups (areas of a repeated mission) and the type of coordinate system used by the
database. They save a NumPy data file for each set of trials; each row in the file
contains the base mission ID, the base mission number, the time (in seconds) to complete
the algorithm, the stage at which the candidate set of legs was empty, and the number of
partner legs it found. The stage of the exit is indicated by the following integer
values:

    1 - no overlapping candidates found in database.
    2 - no candidates met the course difference criteria.
    3 - no candidates met the distance criteria.
    4 - no candidates met the success rate threshold.
    5 - partner legs found.
