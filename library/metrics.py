# SPDX-FileCopyrightText: SAS research group, HFT, Helmut Schmidt University
# SPDX-License-Identifier: CC0-1.0
# https://github.com/hsu-sonar/icua24-geopackage

import numpy as np


def line_piece_distances(base, partners, return_all=False):
    """Calculate line piece distances between a base segment and some partner segments.

    Note that the shape of the output arrays is described relative to the shape of the
    ``partners`` input array.

    See the "Line piece distances" notebook for a derivation of the formulas used to
    find the intermediate point and test for intersections.

    Parameters
    ----------
    base : array-like
        An array [[point0_easting, point0_northing], [point1_easting, point1_northing]]
        specifying the endpoints of the base segment.
    partners : array-like
        An array containing the endpoints of the partner segments. This must be at least
        two dimensional with a shape (..., 2, 2). The last two dimensions contains the
        points in the same order as ``base``.
    return_all : Boolean
        If False, only the line piece minimum and line piece average distances will be
        returned. If True, the distances ``d1, ..., d4`` and the result of the
        intersection test will also be returned.

    Returns
    -------
    line_piece_minimum, line_piece_average : numpy.ndarray
        Arrays of the distances. These will have a shape (...).
    distances : numpy.ndarray
        The distances d1 through d4 used in the calculation. This will have a shape
        (4, ...) with the first axis corresponding to the different distances. It will
        only be returned if ``return_all`` is True.
    intersects : numpy.ndarray
        A Boolean array indicating which of the partner legs intersect the base leg.
        This will have a shape (...). It will only be returned if ``return_all`` is
        True.

    """
    # Ensure it is an array of the correct shape.
    base = np.array(base)
    if base.shape != (2, 2):
        raise ValueError("invalid shape for base coordinates")

    # Ensure it is an array with the last two dimensions being the correct shape.
    partners = np.array(partners)
    if partners.ndim < 2:
        raise ValueError("partner coordinate array must be at least 2d")
    if partners.shape[-2:] != (2, 2):
        raise ValueError("invalid shape for last 2 dimensions of partner coordinates")

    # Reserve space for the four sets of points.
    Pshape = [4] + list(partners.shape)
    del Pshape[-2]
    P = np.empty(Pshape, dtype=float)

    # Take the start and end of the base leg as P1 and P3.
    P[0] = base[0]
    P[2] = base[1]

    # Euclidean distances from P1 to start of partners and P3 to end of partners, and
    # the element-wise minimum (the metric u in the paper).
    distnorm_a = np.linalg.norm(partners[..., 0, :] - P[0], axis=-1)
    distnorm_b = np.linalg.norm(partners[..., 1, :] - P[2], axis=-1)
    distnorm = np.minimum(distnorm_a, distnorm_b)

    # Repeat for the flipped case.
    distflip_a = np.linalg.norm(partners[..., 1, :] - P[0], axis=-1)
    distflip_b = np.linalg.norm(partners[..., 0, :] - P[2], axis=-1)
    distflip = np.minimum(distflip_a, distflip_b)

    # Boolean mask of partners, True where P2 and P4 should be the start and finish of
    # the partner leg (the 'normal' case above).
    mask = distnorm <= distflip

    # Use this to select appropriate points for P2 and P4. The first argument to
    # select() is a list of masks and the second is a list of arrays. The output has
    # elements from array0 where mask0 is True and so on.
    mask = mask[..., np.newaxis]
    P[1] = np.select([mask, ~mask], [partners[..., 0, :], partners[..., 1, :]])
    P[3] = np.select([~mask, mask], [partners[..., 0, :], partners[..., 1, :]])

    # Allocate space for the intermediate distances.
    dshape = list(P.shape)[1:]
    dshape[-1] = 4
    d = np.empty(dshape, dtype=float)

    # Compute each distance.
    for i in range(4):
        # Point indices for the other line.
        A = (i + 1) % 4
        B = (i + 3) % 4

        # Find the parameter t. Note we compute the dot product ourselves rather than
        # using np.dot() as the latter has different behaviours depending on the
        # dimensionality of the inputs.
        dirvec = P[B] - P[A]
        offvec = P[i] - P[A]
        t = np.clip(np.sum(dirvec * offvec, axis=-1) / np.sum(dirvec**2, axis=-1), 0, 1)

        # Calculate the distance between the starting point and the intermediate point.
        H = P[A] + t[..., np.newaxis] * dirvec
        d[..., i] = np.linalg.norm(P[i] - H, axis=-1)

    # Calculate the metrics assuming no intersections.
    dmin = d.min(axis=-1)
    dlpa = 0.5 * (d[..., :2].min(axis=-1) + d[..., 2:].min(axis=-1))

    # These may have collapsed to single values. This ensures they are arrays so we can
    # index them later.
    dmin = np.atleast_1d(dmin)
    dlpa = np.atleast_1d(dlpa)

    # Precompute the three differences needed for the determinants. The naming BA means
    # B - A and so on.
    BA = P[2] - P[0]
    CD = P[1] - P[3]
    CA = P[1] - P[0]

    # Compute the determinants.
    detA = BA[..., 0] * CD[..., 1] - CD[..., 0] * BA[..., 1]
    detAr = CA[..., 0] * CD[..., 1] - CD[..., 0] * CA[..., 1]
    detAs = BA[..., 0] * CA[..., 1] - CA[..., 0] * BA[..., 1]

    # Adjust the signs to simplify the comparison.
    sgn = np.sign(detA)
    detA = np.abs(detA)
    detAr *= sgn
    detAs *= sgn

    # Find which partner legs intersect the base leg.
    intersects = (
        ~np.isclose(detA, 0, rtol=0, atol=1e-9)
        & (0 <= detAr)
        & (detAr <= detA)
        & (0 <= detAs)
        & (detAs <= detA)
    )

    # And adjust the distances accordingly.
    dmin[intersects] = 0
    dlpa[intersects] *= 0.5

    if return_all:
        return dmin, dlpa, d, intersects
    return dmin, dlpa
