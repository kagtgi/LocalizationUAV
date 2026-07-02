"""Step 5b - online query: K-NN retrieval + plurality vote (paper §4.5b).

For each UAV descriptor, retrieve K=5 nearest satellite triangles by ell_1.
Aggregate votes per parent ``patch_id``; the plurality winner determines the
predicted satellite patch, whose pre-computed mean triangle centroid (in
parent-image pixel coordinates) is returned as the predicted position.

``query_uav`` returns only the rank-1 patch (the paper's headline output).
``query_top_n_patches`` returns the top N patches ranked by plurality vote,
which is useful for visual analysis ("the top 100 most plausible patches").
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..database.kdtree import SatelliteDatabase


@dataclass
class PatchPrediction:
    """One ranked candidate patch returned by :func:`query_top_n_patches`."""

    rank: int            # 1 = plurality winner
    patch_id: str
    pixel_xy: Tuple[float, float]
    vote_count: int

    def to_dict(self) -> dict:
        return {
            "rank": int(self.rank),
            "patch_id": str(self.patch_id),
            "pixel_xy": (float(self.pixel_xy[0]), float(self.pixel_xy[1])),
            "vote_count": int(self.vote_count),
        }


@dataclass
class QueryResult:
    patch_id: str
    winner_code: int
    vote_count: int
    second_place_votes: int
    pixel_xy: Tuple[float, float]
    all_votes: Counter
    nearest_distances: np.ndarray
    nearest_indices: np.ndarray
    k: int
    top_n: List[PatchPrediction] = field(default_factory=list)

    @property
    def margin(self) -> int:
        return int(self.vote_count) - int(self.second_place_votes)

    def to_dict(self) -> dict:
        return {
            "patch_id": self.patch_id,
            "winner_code": int(self.winner_code),
            "vote_count": int(self.vote_count),
            "second_place_votes": int(self.second_place_votes),
            "pixel_xy": (float(self.pixel_xy[0]), float(self.pixel_xy[1])),
            "margin": self.margin,
            "all_votes": dict(self.all_votes),
            "k": int(self.k),
            "top_n": [p.to_dict() for p in self.top_n],
        }


def plurality_vote(
    indices: np.ndarray,
    patch_ids: np.ndarray,
) -> Tuple[str, int, int, Counter]:
    """String-array helper (kept for tests and external callers)."""
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
    distances: Optional[np.ndarray] = None,
    max_distance: Optional[float] = None,
) -> Tuple[int, int, int, np.ndarray]:
    """Vote with integer codes.

    Returns ``(winner_code, winner_votes, runner_up, counts)`` where
    ``counts`` is a length-P int32 array. If ``max_distance`` is given (with
    matching ``distances``), a match only casts a vote when its retrieval
    distance is within that bound - distance-gated voting, which rejects
    matches too far to plausibly be the same physical corner rather than
    letting every K-NN slot vote regardless of match quality.
    """
    flat_codes = patch_id_codes[np.asarray(indices).flatten()]
    if max_distance is not None and distances is not None:
        keep = np.asarray(distances, dtype=np.float64).flatten() <= float(max_distance)
        flat_codes = flat_codes[keep]
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
    if runner_up > winner_votes:
        runner_up = winner_votes
    return winner_code, winner_votes, runner_up, counts


def _weighted_vote_codes(
    indices: np.ndarray,
    distances: np.ndarray,
    patch_id_codes: np.ndarray,
    max_distance: Optional[float] = None,
) -> Tuple[int, int, int, np.ndarray]:
    """Distance-weighted voting (ablation): each match votes 1/(1 + d).

    Returns the same shape of result as :func:`_plurality_vote_codes`; the
    per-patch weighted sums are returned as ``counts`` (float64 rounded into
    int semantics is avoided - callers treat counts ordinally). ``max_distance``
    applies the same distance gate as :func:`_plurality_vote_codes`.
    """
    flat_codes = patch_id_codes[np.asarray(indices).flatten()]
    flat_dist = np.asarray(distances, dtype=np.float64).flatten()
    if max_distance is not None:
        keep = flat_dist <= float(max_distance)
        flat_codes = flat_codes[keep]
        flat_dist = flat_dist[keep]
    if flat_codes.size == 0:
        return -1, 0, 0, np.zeros(0, dtype=np.float64)
    n_patches = int(patch_id_codes.max()) + 1 if patch_id_codes.size else 0
    weights = 1.0 / (1.0 + flat_dist)
    counts = np.bincount(flat_codes, weights=weights, minlength=n_patches)
    winner_code = int(counts.argmax())
    winner_votes = float(counts[winner_code])
    if counts.size > 1:
        runner_up = float(np.partition(counts, -2)[-2])
    else:
        runner_up = 0.0
    if runner_up > winner_votes:
        runner_up = winner_votes
    return winner_code, winner_votes, runner_up, counts


def _top_n_from_counts(
    counts: np.ndarray,
    db: SatelliteDatabase,
    n: int,
) -> List[PatchPrediction]:
    """Pick the top ``n`` patch codes by vote count and build PatchPrediction list."""
    if counts.size == 0 or n <= 0:
        return []
    voted_mask = counts > 0
    voted_codes = np.where(voted_mask)[0]
    if voted_codes.size == 0:
        return []
    voted_counts = counts[voted_codes]
    # Sort by descending vote count.
    order = np.argsort(-voted_counts, kind="stable")
    top_codes = voted_codes[order][: int(n)]
    top_counts = voted_counts[order][: int(n)]
    out: List[PatchPrediction] = []
    for rank, (code, votes) in enumerate(zip(top_codes, top_counts), start=1):
        centroid = db.patch_centroids[code]
        out.append(
            PatchPrediction(
                rank=rank,
                patch_id=str(db.patch_id_strings[code]),
                pixel_xy=(float(centroid[0]), float(centroid[1])),
                vote_count=int(votes),
            )
        )
    return out


def query_uav(
    uav_descriptors: np.ndarray,
    db: SatelliteDatabase,
    k: int = 5,
    top_n: int = 100,
    p: float = 1,
    weighted: bool = False,
    max_vote_distance: Optional[float] = None,
) -> Optional[QueryResult]:
    """Retrieve K-NN, vote, return the rank-1 patch centroid plus the top-N list.

    ``top_n`` controls how many ranked patches are returned on the result's
    ``top_n`` field (does NOT change the plurality winner). Set ``top_n=0``
    to skip building this list.

    ``p`` is the Minkowski retrieval metric (1 = ell_1, the paper's choice;
    2 only for the ell_2 ablation). ``weighted=True`` switches plurality
    voting to distance-weighted voting (weight 1/(1+d) per match) for the
    voting-strategy ablation; vote counts are then rounded weighted sums.
    ``max_vote_distance``, if set, gates voting: a K-NN match only casts a
    vote when its retrieval distance is within this bound (see
    :func:`_plurality_vote_codes`).
    """
    uav_descriptors = np.ascontiguousarray(np.asarray(uav_descriptors, dtype=np.float32))
    if uav_descriptors.shape[0] == 0 or db.size == 0:
        return None

    distances, indices = db.query(uav_descriptors, k=int(k), p=p)
    if weighted:
        winner_code, winner_votes, runner_up, counts = _weighted_vote_codes(
            indices, distances, db.patch_id_codes, max_distance=max_vote_distance
        )
    else:
        winner_code, winner_votes, runner_up, counts = _plurality_vote_codes(
            indices, db.patch_id_codes, distances=distances, max_distance=max_vote_distance
        )
    if winner_code < 0:
        return None

    centroid_xy = db.patch_centroids[winner_code]
    winner_id = str(db.patch_id_strings[winner_code])

    voted_mask = counts > 0
    voted_codes = np.where(voted_mask)[0]
    string_counter: Counter = Counter()
    for c in voted_codes:
        string_counter[str(db.patch_id_strings[c])] = int(counts[c])

    top_n_list = _top_n_from_counts(counts, db, n=int(top_n)) if top_n > 0 else []

    return QueryResult(
        patch_id=winner_id,
        winner_code=int(winner_code),
        vote_count=int(winner_votes),
        second_place_votes=int(runner_up),
        pixel_xy=(float(centroid_xy[0]), float(centroid_xy[1])),
        all_votes=string_counter,
        nearest_distances=distances,
        nearest_indices=indices,
        k=int(k),
        top_n=top_n_list,
    )


def query_top_n_patches(
    uav_descriptors: np.ndarray,
    db: SatelliteDatabase,
    n: int = 100,
    k: int = 5,
) -> List[PatchPrediction]:
    """Return the top-``n`` patches by plurality vote, in rank order (1 = winner).

    Convenience wrapper around :func:`query_uav` for callers that only need
    the ranked list.
    """
    result = query_uav(uav_descriptors, db, k=int(k), top_n=int(n))
    return result.top_n if result is not None else []
