# SPDX-FileCopyrightText: SAS research group, HFT, Helmut Schmidt University
# SPDX-License-Identifier: CC0-1.0
# https://github.com/hsu-sonar/icua24-geopackage

"""Functions to generate a GeoPackage with random data.

The data is clustered into mission groups, each of which defines a base profile. Most of
the missions in the group follow this profile, but some will perform a different profile
in the same area. See the `get_default_settings` method for a description of the various
parameters which are used to control the random distributions used to in the generation
process.

In most cases, you want to get a dictionary of default generation settings from
`get_default_settings`, modify it as desired and then call `generate_random_gpkg`.

"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import quaternion
import simplification
from fudgeo import geometry
from fudgeo.enumeration import GeometryType, SQLFieldType
from fudgeo.geopkg import Field, GeoPackage, SpatialReferenceSystem
from pyproj import CRS, Transformer
from tqdm import tqdm as terminal_tqdm
from tqdm.notebook import tqdm as notebook_tqdm


class dummy_tqdm:
    """Dummy progress bar with TQDM interface.

    Used when no visible progress bar is wanted to avoid the need for branches.

    """

    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, tb):
        pass


def get_default_settings():
    """Get the default settings for the GeoPackage generator.

    * seed: positive integer giving the seed for the pseudo-random number generator

    * bounding_boxes: a list of bounding boxes. Each item should be of two tuples of
        (latitude, longitude) pairs in the EPSG:4326 (GPS) coordinate system giving two
        opposite corners of a rectangular area to place mission groups in.

    * CRS: the coordinate reference system (CRS) to use for geospatial coordinate in the
        database. Can be any value which can be given to the initialiser of the
        `pyproj.CRS` class, including an existing CRS instance.

    * groups: a dictionary with information about the mission groups to create:
        * count: integer giving the number of mission groups
        * separation: float giving the minimum separation between the centres of mission
            groups in metres.
        * max_iterations: integer giving the maximum number of iterations when trying to
            find centre positions for all the mission groups.
        * P_circular: float giving the probability that the base profile for a mission
            group is circular.
        * time: a two-tuple of `datetime.datetime` giving the range of start times for
            the base profile of a mission group.

    * missions: a dictionary with information about each mission:
        * mean: the mean number of missions per mission group
        * std: the standard deviation of the number of missions per mission group
        * P_inverse_type: the probability that a mission will be the opposite type
            (circular instead of linear or vice-versa) to the base profile of the
            mission group.
        * P_group_profile: the probability that a mission of the same type will follow
            the base profile of the mission group. Missions which don't will have a
            different number of legs, different positions, different radii etc to the
            base progile.
        * time: the mission start time is offset from the group start time by a number
            of days drawn from a uniform distribution. This setting is a list of
            two-tuples giving the relative frequency and maximum offset in days. The
            default ``[(0.8, 21), (0.2, 365)]`` means that 80% of the missions will be
            within ±21 days and 20% within ±365 days.

    * linear: a dictionary with settings for linear missions:
        * mean_legs: the mean number of legs per linear mission
        * std_legs: the standard deviation of the number of legs in a linear mission
        * length: the (minimum, maximum) length of legs in linear missions. The length
            for each mission will be drawn from a uniform distribution.
        * separation: the (minimum, maximum) lateral separation between legs. The
            separation for each mission will be drawn from a uniform distribution.

    * circular: a dictionary with settings for circular missions:
        * N_circles: the (minimum, maximum) number of circles in a circular mission. The
            number of circles will be drawn from a uniform distribution, and will be
            fractional.
        * radius: the (minimum, maximum) radius of circles in a circular mission. The
            radius of a mission will be drawn from a uniform distribution.

    * simplify: a dictionary giving keyword parameters for the trajectory simplification
        algorithm. See the documentation of `simplification.simplify_trajectory` for
        available parameters.

    * systems: the available systems as a tuple of (frequency, details). The first value
        gives the relative frequency of the system being picked; the default values will
        use the "Basic system 1" 75% of the time and the "Dual frequency system" 25% of
        the time. The details are a dictionary with the following entries:
        * name: the system name as saved in the mission table
        * velocity: the velocity the system moves at
        * sample_spacing: the INS sample spacing in metres
        * wobble_std: the standard deviation of the zero-mean normal distribution used
            to generate the random walk for the side-to-side wobble of the system.
        * sensors: a list of (name, frequency, [minimum range, maximum range]) tuples
            giving details of each SAS sensor attached to the system.

    Returns
    -------
    dict

    """
    return {
        "seed": 171716,
        "bounding_boxes": [[(57.447, 6.125), (54.136, 7.590)]],
        "CRS": 4326,
        "groups": {
            "count": 5,
            "separation": 2000.0,
            "max_iterations": 20,
            "P_circular": 0.5,
            "time": (datetime(2020, 6, 30, 8, 0, 0), datetime(2023, 4, 30, 16, 0, 0)),
        },
        "missions": {
            "mean": 4,
            "std": 2,
            "P_inverse_type": 0.2,
            "P_group_profile": 0.9,
            "time": [
                (0.8, 21),
                (0.2, 365),
            ],
        },
        "linear": {
            "mean_legs": 6,
            "std_legs": 4,
            "length": (200, 1800),
            "separation": (40, 100),
        },
        "circular": {
            "N_circles": (1, 5),
            "radius": (30, 70),
        },
        "simplify": {
            "L_max": 50.0,
            "L_min": 10.0,
            "max_err": 0.5,
            "alpha": 0.8,
        },
        "systems": [
            (
                0.75,
                {
                    "name": "Basic system 1",
                    "velocity": 1.5,
                    "sample_spacing": 0.12,
                    "wobble_std": 0.06,
                    "sensors": [
                        ["Sensor 1", 150e3, [5, 75]],
                    ],
                },
            ),
            (
                0.25,
                {
                    "name": "Dual frequency system",
                    "velocity": 1.5,
                    "sample_spacing": 0.12,
                    "wobble_std": 0.06,
                    "sensors": [
                        ["Sensor 2", 100e3, [5, 75]],
                        ["Sensor 3", 300e3, [3, 60]],
                    ],
                },
            ),
        ],
    }


def generate_random_gpkg(filename, settings=None, overwrite=False, progress=None):
    """Generate a GeoPackage database with random entries.

    Parameters
    ----------
    filename : path-like
        The filename of the output GeoPackage.
    settings : dict
        Generation settings in the format returned by `get_default_settings`. If None,
        the default settings will be used.
    overwrite : Boolean
        If True, overwrite any existing GeoPackage at the output filename. If False,
        raise an error if the GeoPackage already exists.
    progress : {None, "terminal", "notebook"}
        Whether to hide the progress bar, or to show one optimised for use in a terminal
        or a Jupyter notebook.

    """
    # Get the appropriate progress bar class.
    if progress is None:
        progress_cls = dummy_tqdm
    elif progress == "terminal":
        progress_cls = terminal_tqdm
    elif progress == "notebook":
        progress_cls = notebook_tqdm
    else:
        raise ValueError(f"unknown value for progress '{progress}'")

    # Deal with an existing output path.
    filename = Path(filename)
    if filename.exists():
        if overwrite:
            filename.unlink()
        else:
            raise FileExistsError(filename)

    if settings is None:
        settings = get_default_settings()

    # Initialise the database.
    db_crs = CRS(settings["CRS"])
    gpkg, srs = initialise_gpkg(filename, db_crs)

    # Process the desired bounding boxes.
    N_bbox = len(settings["bounding_boxes"])
    bboxes_emin = np.empty(N_bbox)
    bboxes_emax = np.empty(N_bbox)
    bboxes_nmin = np.empty(N_bbox)
    bboxes_nmax = np.empty(N_bbox)
    bboxes_area = np.empty(N_bbox)
    db_transforms = np.empty(N_bbox, dtype=object)
    for i, bbox in enumerate(settings["bounding_boxes"]):
        # Determine a suitable projection for generating the positions by taking the
        # mean of the bounding box corners as the centre of a transverse Mercator
        # projection.
        bbox = np.array(bbox)
        bbox_lat0, bbox_lon0 = np.mean(bbox, axis=0)
        bbox_crs = CRS(f"+proj=tmerc +lat_0={bbox_lat0} +lon_0={bbox_lon0} +axis=ned")

        # Convert the corners to eastings and northings in this projection.
        bbox_e, bbox_n = Transformer.from_crs(4326, bbox_crs, always_xy=True).transform(
            bbox[:, 1], bbox[:, 0]
        )
        bboxes_nmax[i], bboxes_nmin[i] = np.max(bbox_n), np.min(bbox_n)
        bboxes_emax[i], bboxes_emin[i] = np.max(bbox_e), np.min(bbox_e)

        # Calculate the area of the bounding box.
        bboxes_area[i] = (bboxes_nmax[i] - bboxes_nmin[i]) * (
            bboxes_emax[i] - bboxes_emin[i]
        )

        # A transformer to move from the coordinate system we will generate positions in
        # to the desired database coordinate system.
        db_transforms[i] = Transformer.from_crs(bbox_crs, db_crs, always_xy=True)

    # Calculate the total area of all bounding boxes, and scale the individual areas to
    # get the probability of one being selected.
    bbox_total_area = np.sum(bboxes_area)
    bboxes_area /= bbox_total_area

    # Initialise our random number generator.
    rng = np.random.default_rng(settings["seed"])

    # Find centre positions for each group.
    N_groups = settings["groups"]["count"]
    group_centres = np.empty((N_groups, 3))
    invalid = np.ones(N_groups, dtype=bool)
    mask_indices = np.triu_indices(N_groups)
    group_transforms = np.empty(N_groups, dtype=object)
    for i in range(settings["groups"]["max_iterations"]):
        # Select a bounding box for each currently invalid group.
        N_needed = invalid.sum()
        bbox_idx = rng.choice(N_bbox, size=N_needed, p=bboxes_area)

        # Get the bounds for each group.
        bbox_emin = bboxes_emin[bbox_idx]
        bbox_emax = bboxes_emax[bbox_idx]
        bbox_nmin = bboxes_nmin[bbox_idx]
        bbox_nmax = bboxes_nmax[bbox_idx]

        # Generate a position for the groups.
        group_centres[invalid, 0] = rng.uniform(bbox_emin, bbox_emax)
        group_centres[invalid, 1] = rng.uniform(bbox_nmin, bbox_nmax)
        group_centres[invalid, 2] = 0.0

        # Store the coordinate transform for each group.
        group_transforms[invalid] = db_transforms[bbox_idx]

        # Calculate the distance between all pairs of points.
        d = np.sqrt(
            ((group_centres - group_centres[:, np.newaxis, :]) ** 2).sum(axis=-1)
        )

        # Set the diagonal and upper triangle of the distance matrix to infinity. The
        # diagonal is the distance from the point to itself, and excluding the upper
        # triangle means we only consider one point of the pair invalid.
        d[mask_indices] = np.inf

        # See if any groups are too close to others.
        invalid = d.min(axis=-1) < settings["groups"]["separation"]
        if not invalid.any():
            break

    if invalid.any():
        raise RuntimeError("could not find suitable group centres")

    # Decide whether the group is circular or linear.
    group_circular = rng.uniform(0, 1, N_groups) < settings["groups"]["P_circular"]

    # Choose the number of missions in each group.`
    N_missions = np.round(
        rng.normal(settings["missions"]["mean"], settings["missions"]["std"], N_groups)
    ).astype(int)
    N_missions[N_missions < 1] = 1
    N_max = N_missions.max()

    # We will now choose parameters for each mission. Since there are a different number
    # of missions in each group, we could store each as a list of arrays, but for
    # simplicity will use a square matrix. This mask allows selection of just the valid
    # components of each parameter.
    mission_mask = (
        np.tile(np.arange(N_max), N_groups).reshape(N_groups, N_max)
        < N_missions[:, np.newaxis]
    )

    # Decide which missions should be the inverse of the group type.
    mission_inverse = (
        rng.uniform(0, 1, (N_groups, N_max)) < settings["missions"]["P_inverse_type"]
    )

    # From this, determine which missions are circular and which are linear.
    mission_circular = np.repeat(group_circular, N_max).reshape(mission_inverse.shape)
    mission_circular[mission_inverse] = ~mission_circular[mission_inverse]

    # Number of legs in each group.
    group_legs = np.round(
        rng.normal(
            settings["linear"]["mean_legs"], settings["linear"]["std_legs"], N_groups
        )
    ).astype(int)
    group_legs[group_legs < 1] = 1
    group_legs[group_circular] = 1

    # Expand this as an initial legs matrix.
    mission_legs = np.repeat(group_legs, N_max).reshape(N_groups, N_max)

    # Override linear groups with an inverse (i.e., circular) mission.
    mission_legs[~group_circular[:, np.newaxis] & mission_inverse] = 1

    # Override circular groups with an inverse (i.e., linear) mission.
    mask = group_circular[:, np.newaxis] & mission_inverse
    mask_legs = np.round(
        rng.normal(
            settings["linear"]["mean_legs"], settings["linear"]["std_legs"], mask.sum()
        )
    ).astype(int)
    mask_legs[mask_legs < 1] = 1
    mission_legs[mask] = mask_legs

    # Decide which missions follow the group profile, and which a different profile.
    mission_same = (
        rng.uniform(0, 1, (N_groups, N_max)) < settings["missions"]["P_group_profile"]
    )

    # Generate new leg counts for linear missions with a different profile.
    mask = ~mission_circular & ~mission_same
    mask_legs = np.round(
        rng.normal(
            settings["linear"]["mean_legs"], settings["linear"]["std_legs"], mask.sum()
        )
    ).astype(int)
    mask_legs[mask_legs < 1] = 1
    mission_legs[mask] = mask_legs

    # Total number of legs to generate.
    N_legs = mission_legs[mission_mask].sum()

    # Start the progress bar and generate each mission group.
    with progress_cls(desc="Legs generated", total=N_legs) as progress_bar:
        for i in range(N_groups):
            gmask = mission_mask[i]
            generate_group(
                settings,
                gpkg,
                srs,
                group_transforms[i],
                rng,
                progress_bar,
                i,
                group_centres[i],
                group_circular[i],
                group_legs[i],
                mission_circular[i, gmask],
                mission_legs[i, gmask],
                mission_same[i],
            )


def initialise_gpkg(filename, crs):
    # Create the initial database.
    gpkg = GeoPackage.create(filename, flavor="EPSG")

    # Database will contain EPSG:4326 as required by the standard. If that is our
    # working CRS, retrieve it.
    if crs.to_epsg() == 4326:
        with gpkg.connection as con:
            cursor = con.execute(
                """SELECT srs_name, organization, organization_coordsys_id, definition,
                          description, srs_id
                   FROM gpkg_spatial_ref_sys
                   WHERE organization='EPSG' and organization_coordsys_id=4326;"""
            )
            srs = SpatialReferenceSystem(*cursor.fetchone())

    # We are using a different working CRS. Add it to the database.
    else:
        # Try to determine an authority and ID which identify the system.
        authority = crs.list_authority()[:1]
        if authority:
            auth_name = authority[0].auth_name
            auth_srid = int(authority[0].code)
        else:
            auth_name = "custom"
            auth_srid = 1

        srs = SpatialReferenceSystem(
            name="dbcoords",
            organization=auth_name,
            org_coord_sys_id=auth_srid,
            definition=crs.to_wkt(),
        )

    # Create the missions table.
    mission_fields = [
        Field("system", SQLFieldType.text),
        Field("mission_name", SQLFieldType.text),
        Field("start_time", SQLFieldType.datetime),
    ]
    gpkg.create_feature_class(
        "missions",
        srs=srs,
        fields=mission_fields,
        shape_type=GeometryType.polygon,
        description="Details of each mission mission",
        spatial_index=True,
    )

    # Rename the primary key and geometry columns.
    with gpkg.connection as con:
        con.execute("ALTER TABLE missions RENAME COLUMN fid to mission_id;")
        con.execute("ALTER TABLE missions RENAME COLUMN SHAPE to mission_area;")
        con.execute(
            "UPDATE gpkg_geometry_columns "
            "SET column_name='mission_area' WHERE table_name='missions';"
        )

    # Create the legs table.
    leg_fields = [
        Field("mission_id", SQLFieldType.integer),
        Field("leg_number", SQLFieldType.integer),
        Field("leg_type", SQLFieldType.text),
        Field("leg_start_time", SQLFieldType.datetime),
        Field("leg_end_time", SQLFieldType.datetime),
    ]
    gpkg.create_feature_class(
        "legs",
        srs=srs,
        fields=leg_fields,
        shape_type=GeometryType.linestring,
        description="Legs followed by the system",
        spatial_index=True,
    )

    # Rename the primary key and geometry columns.
    with gpkg.connection as con:
        con.execute("ALTER TABLE legs RENAME COLUMN fid to leg_id;")
        con.execute("ALTER TABLE legs RENAME COLUMN SHAPE to trajectory;")
        con.execute(
            "UPDATE gpkg_geometry_columns "
            "SET column_name='trajectory' WHERE table_name='legs';"
        )

    # Create the images table.
    image_fields = [
        Field("leg_id", SQLFieldType.integer),
        Field("sensor", SQLFieldType.text),
        Field("frequency", SQLFieldType.float),
        Field("side", SQLFieldType.text),
    ]
    gpkg.create_feature_class(
        "images",
        srs=srs,
        fields=image_fields,
        shape_type=GeometryType.polygon,
        description="Images captured by the system",
        spatial_index=True,
    )

    # Rename the primary key and geometry columns.
    with gpkg.connection as con:
        con.execute("ALTER TABLE images RENAME COLUMN SHAPE to image_area;")
        con.execute("ALTER TABLE images RENAME COLUMN fid to image_id;")
        con.execute(
            "UPDATE gpkg_geometry_columns "
            "SET column_name='image_area' WHERE table_name='images';"
        )

    return gpkg, srs


def circular_mission(
    gpkg,
    srs,
    db_transform,
    rng,
    progress_bar,
    system,
    centre,
    mission_name,
    mission_time,
    N_circles,
    radius,
    offset,
    simplify_params,
):
    """Generate a circular mission.

    Parameters
    ----------
    gpkg : fudgeo.geopkg.GeoPackage
        The database to add the mission to.
    srs : fudgeo.geopkg.SpatialReferenceSystem
        The spatial reference system to use for the mission.
    db_transform : pyproj.Transformer
        A transformer from the Cartesian coordinates used for generation to the desired
        database reference system.
    rng : number.random.Generator
        The pseudo-random number generator to use during generation.
    progress_bar : tqdm.tqdm
        Progress bar instance to update when each leg is complete.
    system : dict
        Details of the system used for the mission.
    centre : numpy.ndarray
        Centre position of the mission group.
    mission_name : str
        Name of the mission.
    mission_time : datetime.datetime
        Date and time that the mission started.
    N_circles : float
        Number of circles in the mission.
    radius : float
        Radius of the mission in metres.
    offset : numpy.ndarray
        Offset centre of the circle relative to the mission group centre.
    simplify_params : dict
        Keyword parameters to pass to `simplification.simplify_trajectory` when
        simplifying the trajectory before inserting it into the database.

    """
    # Decide the direction of rotation and the start angle (from east towards north).
    port = rng.uniform(0, 1) < 0.5
    start_angle = rng.uniform(0, 2 * np.pi)

    # Calculate the circle circumference and the linear distance at each INS point.
    c = 2 * np.pi * radius * N_circles
    L = np.arange(0, c, system["sample_spacing"])

    # Generate the radius at each sample.
    dr = rng.normal(0, system["wobble_std"], len(L))
    r = radius + np.cumsum(dr)

    # Calculate the angle at each INS point.
    angles = np.linspace(0, 2 * N_circles * np.pi, len(L))
    if port:
        angles = start_angle + angles
    else:
        angles = start_angle - angles

    # Turn this information into the Cartesian coordinates.
    leg_e = r * np.cos(angles)
    leg_n = r * np.sin(angles)
    leg_z = np.zeros_like(leg_e)

    # Stack into the full trajectory and then simplify.
    full = np.stack([leg_e, leg_n, leg_z], axis=-1) + centre + offset
    simplified = simplification.simplify_trajectory(full, **simplify_params)

    # Convert into the database coordinate system.
    dbcoords = np.stack(db_transform.transform(*simplified.T), axis=-1)
    linestring = geometry.LineString(dbcoords[:, :2], srs_id=srs.srs_id)

    # Calculate the time of the final sample.
    leg_end = mission_time + timedelta(seconds=c / system["velocity"])

    # And insert the leg into the database.
    with gpkg.connection as con:
        cursor = con.execute(
            """INSERT INTO legs(mission_id, leg_number, leg_type, leg_start_time,
            leg_end_time, trajectory) VALUES (0, ?, 'circular', ?, ?, ?);""",
            [1, mission_time, leg_end, linestring],
        )
        leg_id = cursor.lastrowid

    # Calculate the image polygons for this system.
    max_imgr = -np.inf
    for name, frequency, extent in system["sensors"]:
        # Radius of the full view area.
        img_r = min(radius - extent[0], extent[1] - radius)
        if img_r <= 0:
            continue
        max_imgr = max(max_imgr, img_r)

        # Sample at 16 points.
        img_angle = np.arange(0, 2 * np.pi, np.pi / 8)
        poly_e = img_r * np.cos(img_angle)
        poly_n = img_r * np.sin(img_angle)
        poly_z = np.zeros_like(poly_e)

        # Stack into a polygon and convert to database coordinates.
        img_coords = np.stack([poly_e, poly_n, poly_z], axis=-1) + centre + offset
        dbcoords = np.stack(db_transform.transform(*img_coords.T), axis=-1)

        # Create a polygon (note this takes a list of sets of coordinates, one for each
        # ring) and insert into the database.
        poly = geometry.Polygon([dbcoords[:, :2]], srs_id=srs.srs_id)
        with gpkg.connection as con:
            con.execute(
                """INSERT INTO images(leg_id, sensor, frequency, side, image_area)
                VALUES(?, ?, ?, ?, ?)""",
                [leg_id, name, frequency, "port" if port else "stbd", poly],
            )

    # Calculate a mission polygon and insert it into the database.
    if max_imgr < 0:
        max_imgr = radius
    poly_e = [-max_imgr, max_imgr, max_imgr, -max_imgr]
    poly_n = [-max_imgr, -max_imgr, max_imgr, max_imgr]
    poly_z = np.zeros_like(poly_e)
    poly_coords = np.stack([poly_e, poly_n, poly_z], axis=-1) + centre + offset
    dbcoords = np.stack(db_transform.transform(*poly_coords.T), axis=-1)
    poly = geometry.Polygon([dbcoords[:, :2]], srs_id=srs.srs_id)
    with gpkg.connection as con:
        cursor = con.execute(
            """INSERT INTO missions(mission_name, system, start_time, mission_area)
            VALUES(?, ?, ?, ?);""",
            [mission_name, system["name"], mission_time, poly],
        )
        mission_id = cursor.lastrowid

    # Update the leg to point to this mission.
    with gpkg.connection as con:
        con.execute(
            "UPDATE legs SET mission_id=? WHERE leg_id=?;", [mission_id, leg_id]
        )

    # Another leg done.
    progress_bar.update()


def linear_mission(
    gpkg,
    srs,
    db_transform,
    rng,
    progress_bar,
    system,
    centre,
    mission_name,
    mission_time,
    startx,
    starty,
    heading,
    length,
    simplify_params,
):
    """Generate a linear mission.

    Parameters
    ----------
    gpkg : fudgeo.geopkg.GeoPackage
        The database to add the mission to.
    srs : fudgeo.geopkg.SpatialReferenceSystem
        The spatial reference system to use for the mission.
    db_transform : pyproj.Transformer
        A transformer from the Cartesian coordinates used for generation to the desired
        database reference system.
    rng : number.random.Generator
        The pseudo-random number generator to use during generation.
    progress_bar : tqdm.tqdm
        Progress bar instance to update when each leg is complete.
    system : dict
        Details of the system used for the mission.
    centre : numpy.ndarray
        Centre position of the mission group.
    mission_name : str
        Name of the mission.
    mission_time : datetime.datetime
        Date and time that the mission started.
    startx, starty : numpy.ndarray
        The starting x and y coordinates of each leg.
    heading : float
        Mission heading.
    length : float
        Length of the legs.
    simplify_params : dict
        Keyword parameters to pass to `simplification.simplify_trajectory` when
        simplifying the trajectory before inserting it into the database.

    """
    # Bounding box of all images in this mission.
    mission_minx = np.inf
    mission_maxx = -np.inf
    mission_miny = np.inf
    mission_maxy = -np.inf

    # Quaternion to rotate from along-/across-track to the desired mission heading.
    q = quaternion.from_rotation_vector([0, 0, heading])

    # Process each leg.
    leg_ids = []
    leg_start = mission_time
    for i in range(len(startx)):
        # Along-track position of each sample.
        if startx[i] < 0:
            leg_x = startx[i] + np.arange(0, length, system["sample_spacing"])
        else:
            leg_x = startx[i] - np.arange(0, length, system["sample_spacing"])
        leg_x += rng.normal(0, 20 * system["wobble_std"])

        # Calculate the across-track position as a random walk.
        leg_dy = rng.normal(0, system["wobble_std"], len(leg_x))
        leg_y = np.cumsum(leg_dy) + starty[i] + rng.normal(0, 6 * system["wobble_std"])

        # Stack into a trajectory and rotate to eastings & northings.
        leg_z = np.zeros_like(leg_x)
        full = np.stack([leg_y, leg_x, leg_z], axis=-1)
        full = quaternion.rotate_vectors(q, full) + centre

        # Simplify and convert to database coordinates.
        simplified = simplification.simplify_trajectory(full, **simplify_params)
        dbcoords = np.stack(db_transform.transform(*simplified.T), axis=-1)
        linestring = geometry.LineString(dbcoords[:, :2], srs_id=srs.srs_id)

        # Calculate time of last sample in the leg.
        leg_end = leg_start + timedelta(seconds=length / system["velocity"])

        # Insert into the database.
        with gpkg.connection as con:
            cursor = con.execute(
                """
                INSERT INTO legs(
                    mission_id, leg_number, leg_type, leg_start_time, leg_end_time,
                    trajectory
                )
                VALUES (0, ?, 'linear', ?, ?, ?);
                """,
                [i, leg_start, leg_end, linestring],
            )
            leg_id = cursor.lastrowid
            leg_ids.append(leg_id)

        # Calculate the image polygons for this system.
        for name, frequency, extent in system["sensors"]:
            # Ensure they are counter-clockwise.
            if startx[i] < 0:
                port = np.array(
                    [
                        [leg_y[0] - extent[0], leg_x[0], leg_z[0]],
                        [leg_y[-1] - extent[0], leg_x[-1], leg_z[-1]],
                        [leg_y[-1] - extent[1], leg_x[-1], leg_z[-1]],
                        [leg_y[0] - extent[1], leg_x[0], leg_z[0]],
                    ]
                )
                stbd = np.array(
                    [
                        [leg_y[0] + extent[0], leg_x[0], leg_z[0]],
                        [leg_y[0] + extent[1], leg_x[0], leg_z[0]],
                        [leg_y[-1] + extent[1], leg_x[-1], leg_z[-1]],
                        [leg_y[-1] + extent[0], leg_x[-1], leg_z[-1]],
                    ]
                )

            else:
                port = np.array(
                    [
                        [leg_y[0] + extent[0], leg_x[0], leg_z[0]],
                        [leg_y[-1] + extent[0], leg_x[-1], leg_z[-1]],
                        [leg_y[-1] + extent[1], leg_x[-1], leg_z[-1]],
                        [leg_y[0] + extent[1], leg_x[0], leg_z[0]],
                    ]
                )
                stbd = np.array(
                    [
                        [leg_y[0] - extent[0], leg_x[0], leg_z[0]],
                        [leg_y[0] - extent[1], leg_x[0], leg_z[0]],
                        [leg_y[-1] - extent[1], leg_x[-1], leg_z[-1]],
                        [leg_y[-1] - extent[0], leg_x[-1], leg_z[-1]],
                    ]
                )

            # Update the mission bounding box.
            mission_minx = min(mission_minx, np.min(port[:, 1]), np.min(stbd[:, 1]))
            mission_maxx = max(mission_maxx, np.max(port[:, 1]), np.max(stbd[:, 1]))
            mission_miny = min(mission_miny, np.min(port[:, 0]), np.min(stbd[:, 0]))
            mission_maxy = max(mission_maxy, np.max(port[:, 0]), np.max(stbd[:, 0]))

            # Rotate to the mission heading and shift to the group location.
            port = quaternion.rotate_vectors(q, port) + centre
            stbd = quaternion.rotate_vectors(q, stbd) + centre

            # Convert to the database CRS and format into polygons.
            port_area = geometry.Polygon(
                [np.stack(db_transform.transform(*port[:, :2].T), axis=-1)],
                srs_id=srs.srs_id,
            )
            stbd_area = geometry.Polygon(
                [np.stack(db_transform.transform(*stbd[:, :2].T), axis=-1)],
                srs_id=srs.srs_id,
            )

            # And insert.
            with gpkg.connection as con:
                con.execute(
                    """INSERT INTO images(leg_id, sensor, frequency, side, image_area)
                    VALUES(?, ?, ?, 'port', ?)""",
                    [leg_id, name, frequency, port_area],
                )
                con.execute(
                    """INSERT INTO images(leg_id, sensor, frequency, side, image_area)
                    VALUES(?, ?, ?, 'starboard', ?)""",
                    [leg_id, name, frequency, stbd_area],
                )

        # Allow some turning time before the next leg.
        if i < len(startx) - 1:
            turn_d = starty[i + 1] - starty[i]
            c = np.pi * turn_d / 2
            leg_start = leg_end + timedelta(seconds=c / system["velocity"])

        # Another leg done.
        progress_bar.update()

    # Generate the mission bounding box.
    bbox = np.array(
        [
            [mission_miny, mission_minx, 0],
            [mission_maxy, mission_minx, 0],
            [mission_maxy, mission_maxx, 0],
            [mission_miny, mission_maxx, 0],
        ]
    )
    bbox = quaternion.rotate_vectors(q, bbox) + centre
    mission_area = geometry.Polygon(
        [np.stack(db_transform.transform(*bbox[:, :2].T), axis=-1)],
        srs_id=srs.srs_id,
    )

    # Insert the mission.
    with gpkg.connection as con:
        cursor = con.execute(
            """INSERT INTO missions(mission_name, system, start_time, mission_area)
            VALUES(?, ?, ?, ?);""",
            [mission_name, system["name"], mission_time, mission_area],
        )
        mission_id = cursor.lastrowid

    # Update all legs to point to this mission.
    with gpkg.connection as con:
        con.executemany(
            "UPDATE legs SET mission_id=? WHERE leg_id=?;",
            zip([mission_id] * len(leg_ids), leg_ids),
        )


def _circular_params(rng, settings, include_offset):
    """Generate parameters for a circular mission.

    Parameters
    ----------
    rng : numpy.random.Generator
        The pseudo-random number generator to use during generation.
    settings : dict
        Settings in the format returned by `get_default_settings`.
    include_offset : Boolean
        If True, add an offset for the mission from the group centre.

    Returns
    -------
    dict

    """
    radius = rng.uniform(*settings["circular"]["radius"])
    N_circles = rng.uniform(*settings["circular"]["N_circles"])
    if include_offset:
        Rmax = settings["circular"]["radius"][1]
        offset = rng.uniform(-Rmax, Rmax, 3)
        offset[2] = 0
    else:
        offset = np.array([0.0, 0.0, 0.0])

    return dict(
        radius=radius,
        N_circles=N_circles,
        offset=offset,
    )


def _linear_params(rng, settings, N_legs):
    """Generate parameters for a linear mission.

    Parameters
    ----------
    rng : numpy.random.Generator
        The pseudo-random number generator to use during generation.
    settings : dict
        Settings in the format returned by `get_default_settings`.
    N_legs : int
        Number of legs in the mission.

    Returns
    -------
    dict

    """
    separation = rng.uniform(*settings["linear"]["separation"])
    length = rng.uniform(*settings["linear"]["length"])
    return dict(
        heading=rng.uniform(0, 2 * np.pi),
        length=length,
        startx=(np.arange(N_legs) % 2 * 2 - 1) * (length / 2),
        starty=separation * (np.arange(N_legs) - (N_legs // 2)),
    )


def generate_group(
    settings,
    gpkg,
    srs,
    db_transform,
    rng,
    progress_bar,
    group_number,
    centre,
    circular,
    legs,
    mission_circular,
    mission_legs,
    mission_same,
):
    """Generate a group of missions.

    Parameters
    ----------
    settings : dict
        Settings in the format returned by `get_default_settings`.
    gpkg : fudgeo.geopkg.GeoPackage
        The database to add the group to.
    srs : fudgeo.geopkg.SpatialReferenceSystem
        The spatial reference system to use for the group.
    db_transform : pyproj.Transformer
        A transformer from the Cartesian coordinates used for generation to the desired
        database reference system.
    rng : number.random.Generator
        The pseudo-random number generator to use during generation.
    progress_bar : tqdm.tqdm
        Progress bar instance to update when each leg is complete.
    group_number : int
        A unique group number.
    centre : numpy.ndarray
        Centre position of the mission group.
    circular : bool
        If the standard mission profile for this group is circular.
    legs : int
        Number of legs in the standard mission profile for this group.
    mission_circular : numpy.ndarray
        Booleans for each mission in the group indicating whether it is circular.
    mission_legs : numpy.ndarray
        Number of legs for each mission in the group.
    mission_same : numpy.ndarray
        Booleans indicating whether each mission in the group follows the standard
        profile.

    """
    # Common settings for every mission./
    common = dict(
        gpkg=gpkg,
        srs=srs,
        db_transform=db_transform,
        rng=rng,
        progress_bar=progress_bar,
        centre=centre,
        simplify_params=settings["simplify"],
    )

    # Cumulative probabilities for the system and relative mission time choices.
    sys_cumulative = np.cumsum([system[0] for system in settings["systems"]])
    time_cumulative = np.cumsum([time[0] for time in settings["missions"]["time"]])

    # General group time.
    start, end = settings["groups"]["time"]
    group_time = start + timedelta(days=rng.uniform(0, (end - start).days))

    # Choose standard group details.
    if circular:
        std_params = _circular_params(rng, settings, False)
    else:
        std_params = _linear_params(rng, settings, legs)

    # And process each mission in the group.
    for i in range(len(mission_circular)):
        # Set a name.
        common["mission_name"] = f"Generated mission {group_number}.{i}"

        # Select a mission time.
        timeval = rng.uniform(0, time_cumulative[-1])
        timeidx = np.where(timeval < time_cumulative)[0][0]
        timeoffset = settings["missions"]["time"][timeidx][1]
        mission_time = group_time + timedelta(days=rng.uniform(-timeoffset, timeoffset))
        common["mission_time"] = mission_time

        # Choose a system.
        sysval = rng.uniform(0, sys_cumulative[-1])
        sysidx = np.where(sysval < sys_cumulative)[0][0]
        system = settings["systems"][sysidx][1]
        common["system"] = system

        # Group is circular.
        if circular:
            # Mission is also circular.
            if mission_circular[i]:
                # Mission follows the group profile.
                if mission_same[i]:
                    circular_mission(**common, **std_params)

                # Mission has a different profile.
                else:
                    diff_params = _circular_params(rng, settings, True)
                    circular_mission(**common, **diff_params)

            # Mission is linear.
            else:
                diff_params = _linear_params(rng, settings, mission_legs[i])
                linear_mission(**common, **diff_params)

        # Group is linear.
        else:
            # Mission is circular.
            if mission_circular[i]:
                diff_params = _circular_params(rng, settings, True)
                circular_mission(**common, **diff_params)

            # Mission is also linear.
            else:
                # Mission follows the group profile.
                if mission_same[i]:
                    linear_mission(**common, **std_params)

                # Has a different profile.
                else:
                    diff_params = _linear_params(rng, settings, mission_legs[i])
                    linear_mission(**common, **diff_params)
