"""Step 5b - online query: K-NN retrieval + plurality vote (paper §4.5b).

For each UAV descriptor, retrieve K=5 nearest satellite triangles by ell_1.
Aggregate votes per parent ``patch_id``; the plurality winner determines the
predicted satellite patch, whose pre-computed mean triangle centroid (in
parent-image pixel coordinates) is returned as the predicted position.

The voting and centroid lookup operate on int patch codes, so no full
N-length patch-id string array is materialized on a query path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..database.kdtree import SatelliteDatabase


@dataclass
class QueryResult:
    patch_id: str
    vote_count: int
    second_place_votes: int
    pixel_xy: Tuple[float, float]
    all_votes: Counter
    nearest_distances: np.ndarray
    nearest_indices: np.ndarray
    k: int

    @property
    def margin(self) -> int:
        return int(self.vote_count) - int(self.second_place_votes)

    def to_dict(self) -> dict:
        return {
            "patch_id": self.patch_id,
            "vote_count": int(self.vote_count),
            "second_place_votes": int(self.second_place_votes),
            "pixel_xy": (float(self.pixel_xy[0]), float(self.pixel_xy[1])),
            "margin": self.margin,
            "all_votes": dict(self.all_votes),
            "k": int(self.k),
        }


def plurality_vote(
    indices: np.ndarray,
    patch_ids: np.ndarray,
) -> Tuple[str, int, int, Counter]:
    """Count votes per patch and return ``(winner, winner_votes, runner_up, counter)``.

    Backward-compatible helper for callers who hold a per-descriptor string
    array. Internal queries prefer :func:`_plurality_vote_codes` (faster).
    """
    flat_patches = patch_ids[np.asarray(indices).flatten()]
    counter: Counter = Counter(flat_patches.tolist())
    if not counter:
        return "", 0, 0, counter
    most_common = counter.most_common(2)
    winner_id, winner_votes = most_common[0]
    runner_up_votes = most_common[1][1] if len(most_common) > 1 else 0
    return str(winner_id), int(winner_votes), int(runner_up_votes), counter


def _plurality_vote_codes(
    indices: np.ndarray,
    patch_id_codes: np.ndarray,
) -> Tuple[int, int, int, np.ndarray]:
    """Vote with integer codes; returns ``(winner_code, winner_votes, runner_up, code_counts)``.

    ``code_counts`` is a 1-D ``int32`` array of length P (the number of unique
    patches), where ``code_counts[c]`` is the number of votes for patch code
    ``c``.
    """
    flat_codes = patch_id_codes[np.asarray(indices).flatten()]
    if flat_codes.size == 0:
        return -1, 0, 0, np.zeros(0, dtype=np.int32)
    n_patches = int(patch_id_codes.max()) + 1 if patch_id_codes.size else 0
    counts = np.bincount(flat_codes, minlength=n_patches).astype(np.int32)
    winner_code = int(counts.argmax())
    winner_votes = int(counts[winner_code])
    if counts.size > 1:
        runner_up = int(np.partition(counts, -2)[-2])
    else:
        runner_up = 0
    # Guard: runner_up should not exceed winner; equality is allowed (tie).
    if runner_up > winner_votes:
        runner_up = winner_votes
    return winner_code, winner_votes, runner_up, counts


def query_uav(
    uav_descriptors: np.ndarray,
    db: SatelliteDatabase,
    k: int = 5,
) -> Optional[QueryResult]:
    """Retrieve K-NN, vote, return the predicted patch centroid pixel.

    Returns ``None`` if either ``uav_descriptors`` is empty or the database
    is empty.
    """
    uav_descriptors = np.ascontiguousarray(np.asarray(uav_descriptors, dtype=np.float32))
    if uav_descriptors.shape[0] == 0 or db.size == 0:
        return None

    distances, indices = db.query(uav_descriptors, k=int(k))
    winner_code, winner_votes, runner_up, counts = _plurality_vote_codes(indices, db.patch_id_codes)
    if winner_code < 0:
        return None

    # O(1) centroid lookup via pre-computed per-patch mean.
    centroid_xy = db.patch_centroids[winner_code]
    winner_id = str(db.patch_id_strings[winner_code])

    # Build a string-keyed Counter only over patches that actually received votes.
    voted_mask = counts > 0
    voted_codes = np.where(voted_mask)[0]
    string_counter: Counter = Counter()
    for c in voted_codes:
        string_counter[str(db.patch_id_strings[c])] = int(counts[c])

    return QueryResult(
        patch_id=winner_id,
        vote_count=int(winner_votes),
        second_place_votes=int(runner_up),
        pixel_xy=(float(centroid_xy[0]), float(centroid_xy[1])),
        all_votes=string_counter,
        nearest_distances=distances,
        nearest_indices=indices,
        k=int(k),
    )
