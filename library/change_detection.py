# SPDX-FileCopyrightText: SAS research group, HFT, Helmut Schmidt University
# SPDX-License-Identifier: CC0-1.0
# https://github.com/hsu-sonar/icua24-geopackage

import itertools
import random
import time

import numpy as np
from pyproj import CRS, Transformer

from metrics import line_piece_distances


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


def find_cd_partners(
    gpkg,
    mission_name,
    leg_number,
    course_threshold=4.2,
    distance_threshold=12.0,
    success_rate_estimator=None,
    success_rate_threshold=80.0,
    include_base_mission=False,
    include_cartesian=True,
    return_stage=False,
):
    """Search for change detection partners in a GeoPackage database.

    Parameters
    ----------
    gpkg : fudgeo.GeoPackage
        The GeoPackage database to search in.
    mission_name : str
        The name of the base mission.
    leg_number : int
        The leg number of the base leg.
    course_threshold, distance_threshold : positive
        Upper thresholds (inclusive) for the course difference (degrees) and line piece
        average distance (m). If a threshold is set to None, that filter is disabled
        (but the metric will still be calculated for return).
    success_rate_estimator : callable
        A function to estimate the change detection success rate. This will be given two
        parameters, the course difference (degrees) and line piece average distance (m),
        as NumPy arrays, and must return a NumPy array of the estimated success rate as
        a percentage. If not given, the success rate will not be used for filtering.
    success_rate_threshold : float
        The lower threshold (inclusive) for the success rate. If None, the success rate
        will be calculated for return, but not used for filtering.
    include_base_mission : Boolean
        Whether to consider other legs from the base mission as candidates.
    include_cartesian : Boolean
        If True, the Cartesian projection of the trajectory into the mission coordinate
        system will be included in the returned base and partner dictionaries under the
        key "trajectory_projected".
    return_stage : Boolean
        Include details of which filter stage the function completed at in the output.

    Returns
    -------
    base : dictionary
        Information about the base leg retrieved from the database.
    partners : list of dictionaries
        Each dictionary will have the information retrieved from the database and the
        calculated metrics.
    stage : int
        The stage at which the function exited:
            1 - no overlapping candidates found in database.
            2 - no candidates met the course difference criteria.
            3 - no candidates met the distance criteria.
            4 - no candidates met the success rate threshold.
            5 - partner legs found.
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

    # Find the base leg.
    with gpkg.connection as con:
        cursor = con.cursor()
        cursor.row_factory = _dict_factory
        cursor.execute(
            """SELECT * FROM legs
            INNER JOIN missions ON legs.mission_id=missions.mission_id
            WHERE
            missions.mission_name=? and legs.leg_number=?;""",
            [mission_name, leg_number],
        )
        base = cursor.fetchone()
        if not base:
            raise KeyError("base leg not found")
        if cursor.fetchall():
            raise KeyError("multiple matches for base leg; invalid database")

    # Load the CRS used for the legs table.
    db_crs = CRS(gpkg.feature_classes["legs"].spatial_reference_system.definition)

    # If needed, generate a local projection and transform the base leg.
    if db_crs.is_geographic:
        lon0, lat0 = np.mean(base["mission_area"].rings[0].coordinates, axis=0)
        proj_crs = CRS(f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +axis=ned")
        transformer = Transformer.from_crs(db_crs, proj_crs, always_xy=True)
        base_coords = np.stack(
            [base["trajectory"].coordinates[0], base["trajectory"].coordinates[-1]]
        )
        base_pos = np.stack(transformer.transform(*base_coords.T), axis=-1)

    else:
        base_pos = np.stack(
            [base["trajectory"].coordinates[0], base["trajectory"].coordinates[-1]],
            axis=0,
        )

    # Add projected coordinates to the returned data.
    if include_cartesian:
        base["trajectory_projected"] = base_pos

    # Determine the mission/leg filtering criteria.
    args = []
    if include_base_mission:
        mission_cond = "(missions.mission_id != ? OR legs.leg_number != ?)"
        args.extend([base["mission_id"], base["leg_number"]])
    else:
        mission_cond = "missions.mission_id != ?"
        args.append(base["mission_id"])
    args.append(base["mission_area"])

    # Select legs with intersecting trajectories.
    with gpkg.connection as con:
        cursor = con.cursor()
        cursor.row_factory = _dict_factory
        cursor.execute(
            f"""SELECT * FROM legs
            INNER JOIN missions ON legs.mission_id=missions.mission_id
            WHERE
            {mission_cond}
            AND st_intersects(GeomFromGPB(trajectory), GeomFromGPB(?));
            """,
            args,
        )
        candidates = cursor.fetchall()

    # Return early if none found.
    if not candidates:
        return (base, [], 1) if return_stage else (base, [])

    # Project the coordinates if required.
    if db_crs.is_geographic:
        cand_coords = np.stack(
            [
                np.stack(
                    [c["trajectory"].coordinates[0], c["trajectory"].coordinates[-1]]
                )
                for c in candidates
            ],
            axis=0,
        )
        cand_e, cand_n = transformer.transform(cand_coords[..., 0], cand_coords[..., 1])
        candidate_pos = np.atleast_3d(np.stack([cand_e, cand_n], axis=-1))
    else:
        candidate_pos = np.stack(
            [
                np.stack(
                    [c["trajectory"].coordinates[0], c["trajectory"].coordinates[-1]]
                )
                for c in candidates
            ],
            axis=0,
        )

    # Calculate course difference.
    dbase = base_pos[-1] - base_pos[0]
    base_course = np.arctan2(dbase[0], dbase[1])
    dcand = np.diff(candidate_pos, axis=-2).squeeze(axis=1)
    candidate_course = np.arctan2(dcand[:, 0], dcand[:, 1])
    course_diff = np.degrees(np.abs(base_course - candidate_course))

    # Filter by course difference if desired.
    if course_threshold is not None:
        mask = course_diff <= course_threshold
        candidates = list(itertools.compress(candidates, mask))
        candidate_pos = candidate_pos[mask]
        course_diff = course_diff[mask]

    if not candidates:
        return (base, [], 2) if return_stage else (base, [])

    # Filter by line piece average distance if desired.
    _, lpa = line_piece_distances(base_pos, candidate_pos)
    if distance_threshold is not None:
        mask = lpa <= distance_threshold
        candidates = list(itertools.compress(candidates, mask))
        candidate_pos = candidate_pos[mask]
        course_diff = course_diff[mask]
        lpa = lpa[mask]

    if not candidates:
        return (base, [], 3) if return_stage else (base, [])

    # Filter by expected success rate.
    if success_rate_estimator:
        sr = success_rate_estimator(course_diff, lpa)
        if success_rate_threshold is not None:
            mask = sr >= success_rate_threshold
            candidates = list(itertools.compress(candidates, mask))
            candidate_pos = candidate_pos[mask]
            course_diff = course_diff[mask]
            lpa = lpa[mask]
            sr = sr[mask]
    else:
        sr = [None] * len(lpa)

    if not candidates:
        return (base, [], 4) if return_stage else (base, [])

    # We have filtered out non-matches, only partners left. Add the metrics and possibly
    # the Cartesian coordinates of the trajectories.
    if include_cartesian:
        partners = [
            candidate
            | {
                "absolute_course_difference": course,
                "line_piece_average_distance": dist,
                "success_rate": s,
                "trajectory_projected": pos,
            }
            for candidate, course, dist, s, pos in zip(
                candidates, course_diff, lpa, sr, candidate_pos
            )
        ]
    else:
        partners = [
            candidate
            | {
                "absolute_course_difference": course,
                "line_piece_average_distance": dist,
                "success_rate": s,
            }
            for candidate, course, dist, s, pos in zip(candidates, course_diff, lpa, sr)
        ]

    if return_stage:
        return base, partners, 5
    return base, partners


def measure_runtime(gpkg, N_trials):
    """Measure the runtime of `find_cd_partners`.

    This measures how long the function to complete for a randomly selected base leg.
    Each trial selects (with replacement) a different base leg. This means the same base
    leg may be timed on multiple trials.

    Note that a non-recorded warmup trial is performed first so any extra startup
    overhead is not included in the results.

    Parameters
    ----------
    gpkg : fudgeo.GeoPackage
        The GeoPackage database to use for the measurement.
    N_trials : int
        The number of trials to perform.

    Returns
    -------
    trials : numpy.ndarray
        An Nx5 integer array with one row per trial. The columns are the mission_id and
        leg_number values for the base mission of the trial, the time in ns it took for
        `find_cd_partners` to complete, the stage at which is finished (see its
        docstring for details) and the number of partners it found.

    """
    # Load all legs as potential base legs.
    with gpkg.connection as con:
        cursor = con.cursor()
        cursor.row_factory = None
        cursor.execute(
            """
            SELECT missions.mission_id, missions.mission_name, legs.leg_number
            FROM legs INNER JOIN missions ON legs.mission_id = missions.mission_id;"""
        )
        bases = cursor.fetchall()

    # Perform the required number of trials, plus a warmup one at the start.
    results = np.empty((N_trials, 5), dtype=int)
    for i in range(-1, N_trials):
        mission_id, mission, leg = random.choice(bases)
        start = time.perf_counter_ns()
        base, partners, stage = find_cd_partners(gpkg, mission, leg, return_stage=True)
        stop = time.perf_counter_ns()

        # Don't try to store the warmup.
        if i == -1:
            continue
        results[i] = [mission_id, leg, stop - start, stage, len(partners)]

    return results
