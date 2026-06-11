"""Step 5a - KD-Tree wrapper for satellite descriptors (paper §4.5).

Uses ``scipy.spatial.KDTree`` with ``p=1`` (Minkowski p=1 = cityblock = ell_1)
exactly as specified in the paper. ``leafsize=40``.

Internal storage layout (memory-efficient at scale):

* ``descriptors``       (N, 5)  float32 - the 5-D triangle descriptors
* ``centroids``         (N, 2)  float32 - per-triangle pixel centroids (parent-image coords)
* ``_patch_id_codes``   (N,)    int32   - index into ``_patch_id_strings``
* ``_patch_id_strings`` (P,)    object  - unique patch identifier strings (lookup table)
* ``_patch_centroids``  (P, 2)  float32 - per-patch mean of triangle centroids (precomputed)

The string explosion ``patch_ids`` (length N) is materialized lazily on first
access via the :attr:`patch_ids` property; ``query_uav`` itself works with
codes and never expands the full string array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


class SatelliteDatabase:
    """K-NN-searchable database of 5-D triangle descriptors.

    Construct directly from arrays (or via :meth:`from_dataframe`). The
    KD-Tree is built lazily on first query; subsequent queries reuse it.
    """

    descriptors: np.ndarray
    centroids: np.ndarray
    parent_tif: str
    leaf_size: int

    def __init__(
        self,
        descriptors: np.ndarray,
        centroids: np.ndarray,
        patch_ids: np.ndarray,
        parent_tif: str = "",
        leaf_size: int = 40,
    ):
        descriptors = np.ascontiguousarray(np.asarray(descriptors, dtype=np.float32))
        centroids = np.ascontiguousarray(np.asarray(centroids, dtype=np.float32))
        patch_ids = np.asarray(patch_ids, dtype=object)
        # The paper's descriptor is 5-D; lower-dimensional slices are accepted
        # for the descriptor-component ablations (alpha-only, MFCA-only).
        if descriptors.ndim != 2 or descriptors.shape[1] < 1:
            raise ValueError(f"descriptors must be shape (N, d>=1); got {descriptors.shape}")
        if centroids.shape != (descriptors.shape[0], 2):
            raise ValueError(f"centroids must be shape (N, 2); got {centroids.shape}")
        if patch_ids.shape != (descriptors.shape[0],):
            raise ValueError(f"patch_ids must be shape (N,); got {patch_ids.shape}")

        unique_strings, codes = np.unique(patch_ids, return_inverse=True)
        self.descriptors = descriptors
        self.centroids = centroids
        self._patch_id_codes = codes.astype(np.int32, copy=False)
        self._patch_id_strings = unique_strings  # object array
        self.parent_tif = str(parent_tif)
        self.leaf_size = int(leaf_size)
        self._tree: Optional[KDTree] = None
        self._patch_centroids: Optional[np.ndarray] = None
        self._patch_ids_cache: Optional[np.ndarray] = None
        self._compute_patch_centroids()

    # ------------- internal helpers -------------

    def _compute_patch_centroids(self) -> None:
        n_patches = len(self._patch_id_strings)
        counts = np.bincount(self._patch_id_codes, minlength=n_patches).astype(np.float32)
        counts_safe = np.maximum(counts, 1.0)
        sum_x = np.bincount(self._patch_id_codes, weights=self.centroids[:, 0], minlength=n_patches)
        sum_y = np.bincount(self._patch_id_codes, weights=self.centroids[:, 1], minlength=n_patches)
        self._patch_centroids = np.stack(
            [(sum_x / counts_safe).astype(np.float32), (sum_y / counts_safe).astype(np.float32)],
            axis=1,
        )

    @classmethod
    def _from_storage(
        cls,
        descriptors: np.ndarray,
        centroids: np.ndarray,
        patch_id_codes: np.ndarray,
        patch_id_strings: np.ndarray,
        parent_tif: str,
        leaf_size: int,
    ) -> "SatelliteDatabase":
        """Internal factory that skips the ``np.unique`` recomputation."""
        instance = cls.__new__(cls)
        instance.descriptors = np.ascontiguousarray(np.asarray(descriptors, dtype=np.float32))
        instance.centroids = np.ascontiguousarray(np.asarray(centroids, dtype=np.float32))
        instance._patch_id_codes = np.asarray(patch_id_codes, dtype=np.int32)
        instance._patch_id_strings = np.asarray(patch_id_strings, dtype=object)
        instance.parent_tif = str(parent_tif)
        instance.leaf_size = int(leaf_size)
        instance._tree = None
        instance._patch_centroids = None
        instance._patch_ids_cache = None
        instance._compute_patch_centroids()
        return instance

    # ------------- public properties -------------

    @property
    def size(self) -> int:
        return int(self.descriptors.shape[0])

    @property
    def n_patches(self) -> int:
        return int(self._patch_id_strings.shape[0])

    @property
    def patch_ids(self) -> np.ndarray:
        """Per-triangle patch-id strings, length N. Computed lazily and cached."""
        if self._patch_ids_cache is None:
            self._patch_ids_cache = self._patch_id_strings[self._patch_id_codes]
        return self._patch_ids_cache

    @property
    def patch_id_codes(self) -> np.ndarray:
        """Per-triangle int32 patch-id codes (index into :attr:`patch_id_strings`)."""
        return self._patch_id_codes

    @property
    def patch_id_strings(self) -> np.ndarray:
        """Unique patch-id strings, length P."""
        return self._patch_id_strings

    @property
    def patch_centroids(self) -> np.ndarray:
        """Per-unique-patch mean centroid (P, 2) in parent-image pixel coords."""
        assert self._patch_centroids is not None  # populated in __init__ / _from_storage
        return self._patch_centroids

    @property
    def tree(self) -> KDTree:
        if self._tree is None:
            self._tree = KDTree(self.descriptors, leafsize=int(self.leaf_size))
        return self._tree

    # ------------- factories -------------

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, parent_tif: str = "", leaf_size: int = 40) -> "SatelliteDatabase":
        """Build a database from a :func:`build_satellite_descriptors` DataFrame."""
        required = {"alpha1", "alpha2", "e1", "e2", "e3", "centroid_x", "centroid_y", "patch_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
        descriptors = df[["alpha1", "alpha2", "e1", "e2", "e3"]].to_numpy(dtype=np.float32)
        centroids = df[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
        patch_ids = df["patch_id"].to_numpy(dtype=object)
        if not parent_tif and "parent_tif" in df.columns and len(df):
            parent_tif = str(df["parent_tif"].iloc[0])
        return cls(
            descriptors=descriptors,
            centroids=centroids,
            patch_ids=patch_ids,
            parent_tif=parent_tif,
            leaf_size=leaf_size,
        )

    # ------------- query -------------

    def query(self, uav_descriptors: np.ndarray, k: int = 5, p: float = 1) -> Tuple[np.ndarray, np.ndarray]:
        """K-NN under Minkowski-``p``; returns ``(distances (M, k), indices (M, k))``.

        ``p=1`` (ell_1/cityblock, the paper's metric) is the default; ``p=2``
        is exposed only for the ell_2 ablation row.
        """
        uav_descriptors = np.ascontiguousarray(np.asarray(uav_descriptors, dtype=np.float32))
        dim = int(self.descriptors.shape[1])
        if uav_descriptors.ndim != 2 or uav_descriptors.shape[1] != dim:
            raise ValueError(f"uav_descriptors must be shape (M, {dim}); got {uav_descriptors.shape}")
        k_eff = int(min(k, self.size))
        if k_eff == 0:
            return (
                np.zeros((uav_descriptors.shape[0], 0), dtype=np.float32),
                np.zeros((uav_descriptors.shape[0], 0), dtype=np.int64),
            )
        # p=1 = Minkowski p=1 = ell_1 (cityblock) per paper §4.5.
        distances, indices = self.tree.query(uav_descriptors, k=k_eff, p=p)
        if k_eff == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        return np.asarray(distances, dtype=np.float32), np.asarray(indices, dtype=np.int64)

    # ------------- serialization -------------

    def save(self, path: str) -> str:
        """Persist arrays to ``path`` as ``.npz``; the KD-Tree is rebuilt on load."""
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            descriptors=self.descriptors,
            centroids=self.centroids,
            patch_id_codes=self._patch_id_codes,
            patch_id_strings=self._patch_id_strings,
            parent_tif=np.array(self.parent_tif),
            leaf_size=np.array(self.leaf_size),
        )
        return path

    @classmethod
    def load(cls, path: str) -> "SatelliteDatabase":
        data = np.load(str(path), allow_pickle=True)
        parent_tif = data["parent_tif"].item() if data["parent_tif"].size else ""
        leaf_size = int(data["leaf_size"].item()) if data["leaf_size"].size else 40
        return cls._from_storage(
            descriptors=data["descriptors"],
            centroids=data["centroids"],
            patch_id_codes=data["patch_id_codes"],
            patch_id_strings=data["patch_id_strings"],
            parent_tif=str(parent_tif),
            leaf_size=leaf_size,
        )

    # ------------- nice repr -------------

    def __repr__(self) -> str:
        return (
            f"SatelliteDatabase(size={self.size}, patches={self.n_patches}, "
            f"parent_tif={self.parent_tif!r}, leaf_size={self.leaf_size})"
        )
