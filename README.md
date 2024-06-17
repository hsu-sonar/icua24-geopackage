<!--

SPDX-FileCopyrightText: SAS research group, HFT, Helmut Schmidt University
SPDX-License-Identifier: CC0-1.0

-->

Application-driven partner leg pairing using a geospatial database
==================================================================

This repository contains code accompanying the paper *Application-driven partner leg
pairing using a geospatial database*, authored by Kaushikk V. N., B Bonnett, H
Schmaljohann, R Klemm and T Fickenscher, and presented at the 2024 International
Conference on Underwater Acoustic in Bath, United Kingdom.


License
=======

The code and accompanying documentation is released under the CC0-1.0 license,
equivalent to public domain in most jurisdictions. For clarity, all files in the
repository are tagged with appropriate SPDX identifiers in accordance with version 3.0
of the [REUSE Specification](https://reuse.software).


Contents
========

Library
-------

The `library` directory contains a Python library with modules for generating a sample
GeoPackage database containing random data and for implementing the example applications
from the paper. The `pyproject.toml` file at the top level of the repository allows this
library to be installed into a Python environment with your preferred package manager.
Note that the other code described below will load the library from the repository and
so it is not required to install the library to run that code.


Notebooks
---------

The `notebooks` directory contains a number of Jupyter notebooks. Some of these describe
the development of the library code while others give examples of how to use the
library.


Benchmarks
----------

The `benchmarks` directory contains some scripts to generate figures of the performance
of the example applications in selecting mission data from the database (equivalent to
Figures 4 and 5 in the paper). The `benchmarks/data` directory contains the databases
used to generate the figures in the paper.
