"""FAISS vector index for semantic nearest-neighbor search over habit nodes.

Wraps IndexFlatIP (inner product on L2-normalized vectors = cosine similarity)
with a node_id ↔ internal-index mapping so callers work with string IDs.

Removal rebuilds the index from the surviving vectors. That is O(n) but
acceptable since removals happen off the hot path (decay/eviction/audits),
and flat indexes don't natively support remove_ids.

Precision: vectors are stored at float32 regardless of input precision.
For 384-d unit vectors this introduces ~1e-7 drift per element, well
below typical downstream tolerances.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import faiss
import numpy as np

from cognigraph.exceptions import PersistenceError
from cognigraph.types import EmbeddingVector, NodeId


# Refuse to load sidecar JSON larger than this — guard against memory exhaustion
# from hostile or corrupt files. 64 MB holds ~1M ids with room to spare.
_MAX_SIDECAR_BYTES = 64 * 1024 * 1024


class FAISSIndex:
    """IndexFlatIP-backed semantic search with string-keyed IDs.

    Thread safety: not thread-safe. Callers must synchronize externally.

    Paths passed to save()/load() are treated as trusted. See TODO(W4) —
    if paths may originate from untrusted sources, validate them upstream.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dim = dimension
        self._index = faiss.IndexFlatIP(dimension)
        # Source of truth for vectors (kept alongside the index so we can
        # rebuild on remove). Each row corresponds to self._index_to_id[row].
        self._vectors: list[np.ndarray] = []
        self._index_to_id: list[NodeId] = []
        self._id_to_index: dict[NodeId, int] = {}

    # --- Basic ops ---

    def count(self) -> int:
        # Authoritative count lives in the parallel list; FAISS's ntotal
        # should always match but we prefer the mapping to surface any drift.
        return len(self._index_to_id)

    @property
    def dimension(self) -> int:
        return self._dim

    def close(self) -> None:
        """Release the FAISS index and drop parallel state.

        After close(), the instance behaves like a freshly-constructed empty
        index. Provided for API symmetry with SQLitePersistence and to let
        long-running processes free C++ memory deterministically.
        """
        self._index = faiss.IndexFlatIP(self._dim)
        self._vectors = []
        self._index_to_id = []
        self._id_to_index = {}

    def __enter__(self) -> FAISSIndex:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def add(self, node_id: NodeId, vector: EmbeddingVector) -> None:
        """Add or replace a vector for `node_id`.

        If `node_id` already exists, the existing entry is overwritten
        (removed then re-added) so the mapping stays consistent.
        """
        # TODO(W1): overwrite is O(n) today because IndexFlat is immutable.
        # If add-with-same-id becomes hot, add a tombstone path that defers
        # rebuild until tombstone density crosses a threshold.
        vec = self._to_row(vector)

        if node_id in self._id_to_index:
            self.remove(node_id)

        self._vectors.append(vec[0])
        self._index_to_id.append(node_id)
        self._id_to_index[node_id] = len(self._index_to_id) - 1
        self._index.add(vec)

    def remove(self, node_id: NodeId) -> None:
        """Remove a vector. No-op if `node_id` isn't indexed."""
        if node_id not in self._id_to_index:
            return

        target = self._id_to_index[node_id]
        self._vectors.pop(target)
        self._index_to_id.pop(target)
        # Rebuild the id mapping in one pass — simpler than in-place decrement
        # and eliminates a class of off-by-one bugs.
        self._id_to_index = {
            nid: i for i, nid in enumerate(self._index_to_id)
        }

        self._rebuild_index()

    def search(
        self, query_vector: EmbeddingVector, k: int = 5
    ) -> list[tuple[NodeId, float]]:
        """Return up to k nearest neighbors by cosine similarity, DESC.

        Scores are clamped to [-1.0, 1.0]: float32 inner products of unit
        vectors can overshoot by a few ulps, and downstream code should see
        values that match the mathematical cosine-similarity range.
        """
        if self._index.ntotal == 0:
            return []

        q = self._to_row(query_vector)
        actual_k = min(k, self._index.ntotal)
        scores, indices = self._index.search(q, actual_k)

        result: list[tuple[NodeId, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            clamped = max(-1.0, min(1.0, float(score)))
            result.append((self._index_to_id[idx], clamped))
        return result

    # --- Persistence ---

    def save(self, path: str) -> None:
        """Atomically persist the index and the id mapping.

        Writes both files to temporary siblings first, fsyncs, then
        os.replace()s them into place. A mid-crash can still leave the
        previous snapshot intact — load() will detect a length mismatch
        between the on-disk index and sidecar if one moved but not the
        other, and raise PersistenceError.
        """
        # TODO(W4): path is trusted. Validate upstream if ever user-supplied.
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            sidecar = _sidecar_path(path)

            tmp_index = target.with_name(target.name + ".tmp")
            tmp_sidecar = sidecar.with_name(sidecar.name + ".tmp")

            # 1. Write index to temp + fsync
            faiss.write_index(self._index, str(tmp_index))
            _fsync_file(tmp_index)

            # 2. Write sidecar to temp + fsync
            payload = json.dumps(
                {"dimension": self._dim, "ids": self._index_to_id}
            )
            tmp_sidecar.write_text(payload)
            _fsync_file(tmp_sidecar)

            # 3. Atomically replace. If we crash between these two os.replace
            # calls, load() will detect the resulting length mismatch.
            os.replace(str(tmp_index), str(target))
            os.replace(str(tmp_sidecar), str(sidecar))
        except (OSError, RuntimeError) as e:
            raise PersistenceError(f"FAISSIndex.save failed: {e}") from e

    def load(self, path: str) -> None:
        """Restore the index and id mapping from `path` + sidecar."""
        target = Path(path)
        sidecar = _sidecar_path(path)
        try:
            if not target.exists():
                raise PersistenceError(f"FAISS index file not found: {path}")
            if not sidecar.exists():
                raise PersistenceError(f"FAISS id sidecar not found: {sidecar}")

            sidecar_size = sidecar.stat().st_size
            if sidecar_size > _MAX_SIDECAR_BYTES:
                raise PersistenceError(
                    f"FAISSIndex.load: sidecar too large "
                    f"({sidecar_size} bytes > {_MAX_SIDECAR_BYTES})"
                )

            index = faiss.read_index(str(target))
            meta = json.loads(sidecar.read_text())
        except (OSError, RuntimeError) as e:
            raise PersistenceError(f"FAISSIndex.load failed: {e}") from e
        except json.JSONDecodeError as e:
            raise PersistenceError(f"FAISSIndex.load: corrupt id sidecar: {e}") from e

        dim = meta.get("dimension") if isinstance(meta, dict) else None
        ids = meta.get("ids") if isinstance(meta, dict) else None

        if not isinstance(dim, int) or dim <= 0:
            raise PersistenceError(f"FAISSIndex.load: invalid dimension {dim!r}")
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise PersistenceError("FAISSIndex.load: ids must be a list of strings")
        if index.d != dim:
            raise PersistenceError(
                f"FAISSIndex.load: sidecar dim {dim} != index dim {index.d}"
            )
        if index.ntotal != len(ids):
            raise PersistenceError(
                f"FAISSIndex.load: index/id length mismatch "
                f"({index.ntotal} vs {len(ids)})"
            )

        self._dim = dim
        self._index = index
        self._index_to_id = list(ids)
        self._id_to_index = {nid: i for i, nid in enumerate(self._index_to_id)}
        # Rehydrate the parallel vector list so future remove() can rebuild
        # from the surviving rows.
        if index.ntotal > 0:
            matrix = index.reconstruct_n(0, index.ntotal)
            self._vectors = [
                np.asarray(matrix[i], dtype=np.float32).copy()
                for i in range(index.ntotal)
            ]
        else:
            self._vectors = []

    # --- Internals ---

    def _to_row(self, vector: EmbeddingVector) -> np.ndarray:
        # Force a copy so faiss.normalize_L2 (in-place) can't mutate a
        # caller-owned float32 ndarray. Also shields us from non-contiguous
        # numpy views.
        arr = np.array(vector, dtype=np.float32, copy=True).reshape(1, -1)

        if arr.shape[1] != self._dim:
            raise ValueError(
                f"vector dim mismatch: expected {self._dim}, got {arr.shape[1]}"
            )
        # TODO(W5): reject non-1-D input shapes up front with a clearer error.

        if not np.all(np.isfinite(arr)):
            raise ValueError("vector contains NaN or Inf")
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            raise ValueError("cannot index zero-norm vector")

        faiss.normalize_L2(arr)
        return arr

    def _rebuild_index(self) -> None:
        self._index = faiss.IndexFlatIP(self._dim)
        if self._vectors:
            matrix = np.stack(self._vectors).astype(np.float32)
            self._index.add(matrix)


# --- Module helpers ---


def _sidecar_path(path: str) -> Path:
    """Return the id-sidecar path for a given index path."""
    return Path(str(path) + ".ids.json")


def _fsync_file(p: Path) -> None:
    """fsync a file so os.replace commits a durable state.

    Best-effort: platforms that don't support fsync (or where the file
    descriptor has already been closed) silently skip. The caller still
    gets atomicity from os.replace; fsync only adds durability.
    """
    try:
        fd = os.open(str(p), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass
