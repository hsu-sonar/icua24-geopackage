# SPDX-FileCopyrightText: SAS research group, HFT, Helmut Schmidt University
# SPDX-License-Identifier: CC0-1.0
# https://github.com/hsu-sonar/icua24-geopackage

import numpy as np


def simplify_trajectory(positions, L_max=50.0, L_min=10.0, max_err=0.5, alpha=0.8):
    """Simplify a trajectory.

    See the "simplifying trajectories" notebook for details on the development of this
    code.

    Parameters
    ----------
    positions : numpy.ndarray
        An (N, 2) array of the position of the vehicle at each sample time.
    L_max, L_min : float
        Maximum and minimum desired segment length.
    max_err : float
        Maximum mean squared distance between the points and the proposed segment.
    alpha : float
        Factor by which to reduce the length after each unsuccessful check.

    Returns
    -------
    simplified : numpy.ndarray
        An (M, 2) array of the vertices of the simplified trajectory.

    """
    # Calculate the cumulative distance between points.
    dist = np.zeros(positions.shape[0], dtype=float)
    dist[1:] = np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=-1))

    # Index of the start point of the current segment in the original array. Used to
    # easily index the distance.
    start_ind = 0

    # Start with the trajectory start, and loop until we have finished.
    simplified = [positions[0]]
    while True:
        # Start point for this segment.
        start = simplified[-1]

        # Distance relative to the start point.
        dist_seg = dist - dist[start_ind]

        # Loop until we find a suitable L.
        L = L_max
        while True:
            # Use the distance array to find indices of points which are after the start
            # point but within the desired travel.
            ind = np.where((dist_seg > 0) & (dist_seg < L))[0]

            # Take the last of these as the proposed end point...
            end = positions[ind[-1]]

            # ... and the rest as the points to test with.
            points = positions[ind[:-1]]

            # Find perpendicular points on the proposed line. See the 'line piece
            # distances' notebook for details on the method.
            dirvec = end - start
            offvec = points - start
            t = np.sum(dirvec * offvec, axis=-1) / np.sum(dirvec**2, axis=-1)
            P = start + t[..., np.newaxis] * dirvec

            # Calculate the perpendicular distance from each point.
            d = np.linalg.norm(P - points, axis=-1)

            # And from that, the error metric.
            err = np.mean(d**2)

            # If the error is low enough or we have reached our minimum allowable
            # length, add the end point to the output sequence and quit this loop.
            if (err <= max_err) or np.isclose(L, L_min):
                start_ind = ind[-1]
                simplified.append(end)
                break

            # Otherwise, shrink L.
            L *= alpha
            if L < L_min:
                L = L_min

        # We have completed another segment. If the end of that segment was close to the
        # end of the original trajectory, add the original end to the sequence and our
        # simplification is complete.
        if (dist[-1] - dist[start_ind]) < L_min:
            simplified.append(positions[-1])
            break

    return np.array(simplified)
