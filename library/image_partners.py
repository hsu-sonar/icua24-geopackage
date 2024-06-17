# SPDX-FileCopyrightText: SAS research group, HFT, Helmut Schmidt University
# SPDX-License-Identifier: CC0-1.0
# https://github.com/hsu-sonar/icua24-geopackage

import random
import time
from datetime import timedelta

import numpy as np
from pyproj import CRS, Transformer


def _dict_factory(cursor, row):
    """sqlite row factory that creates a dictionary for each row.

    Parameters
    ----------
    cursor
        The cursor being used to fetch data.
    row
        The current row being loaded.

    Returns
    -------
    dict

    """
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def find_image_partners(
    gpkg,
    image_id,
    same_sensor=False,
    same_frequency=False,
    include_base_mission=True,
    minimum_area=None,
    time_difference=None,
    return_stage=False,
):
    """Find partner images.

    Parameters
    ----------
    gpkg : fudgeo.GeoPackage
        The GeoPackage database to search in.
    image_id : int
        The GeoPackage image ID of the base image.
    same_sensor, same_frequency : Boolean
        If True, limit the candidate images to those captured by the same sensor or at
        the same frequency, respectively.
    include_base_mission : Boolean
        If True, consider other images from the same mission as candidates. If False,
        only images from other missions will be considered.
    minimum_area : float, optional
        The minimum area (in metres squared) of the intersection between the base image
        and a partner leg. If not given, no minimum area is required.
    time_difference : datetime.timedelta, tuple, optional
        The maximum difference in start time between the leg used for the base image and
        the leg used for a partner image. If None, no limits are applied. If a single
        timedelta is given, this is the maximum difference before or after the base
        leg. A tuple `(before, after)` of timedeltas can be given to apply asymmetrical
        limits.

     return_stage : Boolean
        Include details of which filter stage the function completed at in the output.

    Returns
    -------
    base : dictionary
        Information about the base image retrieved from the database.
    partners : list of dictionaries
        Each dictionary will have the information retrieved from the database and the
        calculated metrics.
    stage : int
        The stage at which the function exited:
            1 - no overlapping candidates found in database.
            2 - no candidates met minimum area criteria.
            3 - partner images found.
        This is only returned if ``return_stage`` is True.

    """
    # Ensure we have GeoPackage support.
    with gpkg.connection as con:
        cursor = con.cursor()
        cursor.row_factory = None
        cursor.execute("select HasGeoPackage() as gpkg;")
        has_gpkg = bool(cursor.fetchone()[0])
    if not has_gpkg:
        raise RuntimeError("GeoPackage support not enabled in SpatiaLite")

    # Find the base image.
    with gpkg.connection as con:
        cursor = con.cursor()
        cursor.row_factory = _dict_factory
        cursor.execute(
            """SELECT * FROM images
            INNER JOIN legs on images.leg_id=legs.leg_id
            INNER JOIN missions ON legs.mission_id=missions.mission_id
            WHERE images.image_id=:image_id""",
            dict(image_id=image_id),
        )
        base = cursor.fetchone()
        if not base:
            raise KeyError("base image not found")
        if cursor.fetchall():
            raise KeyError("multiple matches for base image; invalid database")

    # Load the CRS used for the images table.
    db_crs = CRS(gpkg.feature_classes["images"].spatial_reference_system.definition)

    # If needed, generate a local projection.
    if db_crs.is_geographic:
        # Generate a projection transformer.
        lon0, lat0 = np.mean(base["image_area"].rings[0].coordinates, axis=0)
        proj_crs = CRS(f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +axis=ned")
        transformer = Transformer.from_crs(db_crs, proj_crs, always_xy=True)

    # Start a list of selection conditions with a basic intersection test.
    where = ["st_intersects(GeomFromGPB(images.image_area), GeomFromGPB(:baseimg))"]
    params = {"baseimg": base["image_area"]}

    # Determine what parts of the base mission to include in candidates.
    if include_base_mission:
        where.append(
            """(missions.mission_id != :base_mission
               OR legs.leg_number != :base_leg
               OR images.side != :base_side)"""
        )
        params["base_mission"] = base["mission_id"]
        params["base_leg"] = base["leg_number"]
        params["base_side"] = base["side"]
    else:
        where.append("missions.mission_id != :base_mission")
        params["base_mission"] = base["mission_id"]

    # Should we limit to the same sensor and/or frequency?
    if same_sensor:
        where.append("images.sensor=:sensor")
        params["sensor"] = base["sensor"]
    if same_frequency:
        where.append("images.frequency=:frequency")
        params["frequency"] = base["frequency"]

    # Place limits on the leg start time.
    if time_difference is not None:
        if isinstance(time_difference, timedelta):
            before, after = time_difference, time_difference
        else:
            before, after = time_difference

        where.append("legs.leg_start>=:start_time")
        params["start_time"] = base["leg_start"] - before
        where.append("legs.leg_start<=:end_time")
        params["end_time"] = base["leg_start"] + after

    # Select images which overlap, and include the overlap polygon.
    with gpkg.connection as con:
        cursor = con.cursor()
        cursor.row_factory = _dict_factory
        cursor.execute(
            f"""SELECT images.*, legs.*, missions.*,
            AsGPB(Intersection(GeomFromGPB(image_area), GeomFromGPB(:baseimg)))
                as "intersection [Polygon]"
            FROM images
            INNER JOIN legs on images.leg_id=legs.leg_id
            INNER JOIN missions ON legs.mission_id=missions.mission_id
            WHERE {' AND '.join(where)};""",
            params,
        )
        candidates = cursor.fetchall()

    if not candidates:
        return (base, [], 1) if return_stage else (base, [])

    # Project the intersection polygon if required and calculate its area with the
    # shoelace formula.
    for candidate in candidates:
        x, y = candidate["intersection"].rings[0].coordinates.T
        if db_crs.is_geographic:
            x, y = transformer.transform(x, y)
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        candidate["intersection_area"] = area
        candidate["intersection_projected"] = np.stack([x, y])

    # Apply any minimum overlap area.
    if minimum_area is not None:
        candidates = [
            candidate
            for candidate in candidates
            if candidate["intersection_area"] >= minimum_area
        ]
        if not candidates:
            return (base, [], 2) if return_stage else (base, [])

    # Project the candidate trajectory if required and compute view angles and ranges.
    for candidate in candidates:
        x, y = candidate["trajectory"].coordinates.T
        if db_crs.is_geographic:
            x, y = transformer.transform(x, y)
        candidate["trajectory_projected"] = np.stack([x, y])
        candidate["view_angles"] = np.arctan2(y, x)
        candidate["view_ranges"] = np.sqrt(x**2 + y**2)

    return (base, candidates, 3) if return_stage else (base, candidates)


def measure_runtime(gpkg, N_trials, **filters):
    """Measure the runtime of `find_image_partners`.

    This measures how long the function to complete for a randomly selected base image.
    Each trial selects (with replacement) a different base image. This means the same
    base image may be timed on multiple trials.

    Note that a non-recorded warmup trial is performed first so any extra startup
    overhead is not included in the results.

    Parameters
    ----------
    gpkg : fudgeo.GeoPackage
        The GeoPackage database to use for the measurement.
    N_trials : int
        The number of trials to perform.
    **filters
        Keyword arguments to pass to `find_image_partners` to filter the results.

    Returns
    -------
    trials : numpy.ndarray
        An Nx4 integer array with one row per trial. The columns are the image_id of the
        base image of the trial, the time in ns it took for `find_image_partners` to
        complete, the stage at which is finished (see its docstring for details) and the
        number of partners it found.

    """
    # Load all images as potential base images.
    with gpkg.connection as con:
        cursor = con.cursor()
        cursor.row_factory = lambda cursor, row: row[0]
        cursor.execute("SELECT image_id from images;")
        bases = cursor.fetchall()

    # Perform the required number of trials, plus a warmup one at the start.
    results = np.empty((N_trials, 4), dtype=int)
    for i in range(-1, N_trials):
        image_id = random.choice(bases)
        start = time.perf_counter_ns()
        base, partners, stage = find_image_partners(
            gpkg, image_id, return_stage=True, **filters
        )
        stop = time.perf_counter_ns()

        # Don't try to store the warmup.
        if i == -1:
            continue
        results[i] = [image_id, stop - start, stage, len(partners)]

    return results
