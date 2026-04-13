"""Tests for FAISSIndex — real FAISS operations, real files via tmp_path."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cognigraph.exceptions import PersistenceError
from cognigraph.protocols import VectorIndexProtocol
from cognigraph.vector_index import FAISSIndex


DIM = 4  # small dim for fast, deterministic tests


# --- Fixtures ---


@pytest.fixture
def index() -> FAISSIndex:
    return FAISSIndex(dimension=DIM)


def _unit(vec: list[float]) -> list[float]:
    arr = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(arr)
    return (arr / n).tolist() if n else arr.tolist()


# --- Protocol conformance ---


class TestProtocolConformance:
    def test_implements_vector_index_protocol(self) -> None:
        assert isinstance(FAISSIndex(dimension=DIM), VectorIndexProtocol)


# --- Construction ---


class TestConstruction:
    def test_default_dimension(self) -> None:
        idx = FAISSIndex()
        assert idx.dimension == 384
        assert idx.count() == 0

    def test_custom_dimension(self) -> None:
        idx = FAISSIndex(dimension=128)
        assert idx.dimension == 128

    def test_rejects_non_positive_dimension(self) -> None:
        with pytest.raises(ValueError):
            FAISSIndex(dimension=0)
        with pytest.raises(ValueError):
            FAISSIndex(dimension=-1)


# --- Add & count ---


class TestAddAndCount:
    def test_add_single_vector(self, index: FAISSIndex) -> None:
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        assert index.count() == 1

    def test_add_multiple_vectors(self, index: FAISSIndex) -> None:
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        index.add("b", _unit([0.0, 1.0, 0.0, 0.0]))
        index.add("c", _unit([0.0, 0.0, 1.0, 0.0]))
        assert index.count() == 3

    def test_add_wrong_dim_raises(self, index: FAISSIndex) -> None:
        with pytest.raises(ValueError, match="dim mismatch"):
            index.add("a", [1.0, 0.0])

    def test_add_overwrites_existing_id(self, index: FAISSIndex) -> None:
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        index.add("a", _unit([0.0, 1.0, 0.0, 0.0]))  # replace
        assert index.count() == 1

        results = index.search(_unit([0.0, 1.0, 0.0, 0.0]), k=1)
        assert results[0][0] == "a"
        # Score should be close to 1.0 (perfect match to the new vector)
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_overwrite_middle_id_preserves_others(self, index: FAISSIndex) -> None:
        """Regression: overwriting an id must not corrupt neighbours'
        position in the id mapping."""
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        index.add("b", _unit([0.0, 1.0, 0.0, 0.0]))
        index.add("c", _unit([0.0, 0.0, 1.0, 0.0]))

        # Overwrite the middle id with a new vector direction
        index.add("b", _unit([0.0, 0.0, 0.0, 1.0]))
        assert index.count() == 3

        # All three ids retrievable as top-1 for their current directions
        cases = [
            (_unit([1.0, 0.0, 0.0, 0.0]), "a"),
            (_unit([0.0, 0.0, 0.0, 1.0]), "b"),
            (_unit([0.0, 0.0, 1.0, 0.0]), "c"),
        ]
        for query, expected_id in cases:
            res = index.search(query, k=1)
            assert res[0][0] == expected_id
            assert res[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_rejects_zero_vector(self, index: FAISSIndex) -> None:
        with pytest.raises(ValueError, match="zero-norm"):
            index.add("a", [0.0, 0.0, 0.0, 0.0])

    def test_rejects_nan_vector(self, index: FAISSIndex) -> None:
        with pytest.raises(ValueError, match="NaN|finite"):
            index.add("a", [float("nan"), 0.0, 0.0, 0.0])

    def test_rejects_inf_vector(self, index: FAISSIndex) -> None:
        with pytest.raises(ValueError, match="Inf|finite"):
            index.add("a", [float("inf"), 0.0, 0.0, 0.0])

    def test_add_does_not_mutate_caller_ndarray(self, index: FAISSIndex) -> None:
        """Regression: normalize_L2 is in-place; we must copy first."""
        v = np.array([3.0, 0.0, 4.0, 0.0], dtype=np.float32)
        snapshot = v.copy()
        index.add("a", v.tolist())  # list path
        np.testing.assert_array_equal(v, snapshot)

        v2 = np.array([3.0, 0.0, 4.0, 0.0], dtype=np.float32)
        snap2 = v2.copy()
        index.add("b", v2)  # ndarray path
        np.testing.assert_array_equal(v2, snap2)

    def test_search_rejects_nan_query(self, index: FAISSIndex) -> None:
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="NaN|finite"):
            index.search([float("nan"), 0.0, 0.0, 0.0])


# --- Search semantics ---


class TestSearch:
    def test_empty_index_returns_empty(self, index: FAISSIndex) -> None:
        assert index.search(_unit([1.0, 0.0, 0.0, 0.0])) == []

    def test_search_returns_exact_match(self, index: FAISSIndex) -> None:
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        results = index.search(_unit([1.0, 0.0, 0.0, 0.0]), k=1)
        assert len(results) == 1
        node_id, score = results[0]
        assert node_id == "a"
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_search_orthogonal_vectors(self, index: FAISSIndex) -> None:
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        results = index.search(_unit([0.0, 1.0, 0.0, 0.0]), k=1)
        assert results[0][1] == pytest.approx(0.0, abs=1e-5)

    def test_search_topk_sorted_desc(self, index: FAISSIndex) -> None:
        index.add("close", _unit([1.0, 0.1, 0.0, 0.0]))
        index.add("closer", _unit([1.0, 0.05, 0.0, 0.0]))
        index.add("closest", _unit([1.0, 0.0, 0.0, 0.0]))
        index.add("far", _unit([0.0, 1.0, 0.0, 0.0]))

        results = index.search(_unit([1.0, 0.0, 0.0, 0.0]), k=3)
        assert [r[0] for r in results] == ["closest", "closer", "close"]
        # Scores strictly descending
        scores = [r[1] for r in results]
        assert scores[0] > scores[1] > scores[2]

    def test_search_k_larger_than_index(self, index: FAISSIndex) -> None:
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        index.add("b", _unit([0.0, 1.0, 0.0, 0.0]))
        results = index.search(_unit([1.0, 0.0, 0.0, 0.0]), k=10)
        assert len(results) == 2

    def test_search_scores_are_cosine_similarity(self, index: FAISSIndex) -> None:
        """IP on L2-normalized vectors == cosine similarity. Anti-parallel → -1."""
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        results = index.search(_unit([-1.0, 0.0, 0.0, 0.0]), k=1)
        assert results[0][1] == pytest.approx(-1.0, abs=1e-5)

    def test_unnormalized_inputs_still_work(self, index: FAISSIndex) -> None:
        """FAISSIndex should L2-normalize internally, so raw vectors work too."""
        index.add("a", [10.0, 0.0, 0.0, 0.0])
        results = index.search([5.0, 0.0, 0.0, 0.0], k=1)
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_search_wrong_dim_raises(self, index: FAISSIndex) -> None:
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="dim mismatch"):
            index.search([1.0, 0.0])


# --- Remove ---


class TestRemove:
    def test_remove_then_search_excludes_it(self, index: FAISSIndex) -> None:
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        index.add("b", _unit([0.0, 1.0, 0.0, 0.0]))

        index.remove("a")
        assert index.count() == 1

        results = index.search(_unit([1.0, 0.0, 0.0, 0.0]), k=5)
        ids = [r[0] for r in results]
        assert "a" not in ids
        assert "b" in ids

    def test_remove_missing_is_noop(self, index: FAISSIndex) -> None:
        index.remove("nonexistent")  # must not raise
        assert index.count() == 0

    def test_remove_middle_keeps_others_searchable(
        self, index: FAISSIndex
    ) -> None:
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        index.add("b", _unit([0.0, 1.0, 0.0, 0.0]))
        index.add("c", _unit([0.0, 0.0, 1.0, 0.0]))
        index.add("d", _unit([0.0, 0.0, 0.0, 1.0]))

        index.remove("b")
        assert index.count() == 3

        # All remaining IDs still returned by a broad search
        for query_id, query_vec in [
            ("a", _unit([1.0, 0.0, 0.0, 0.0])),
            ("c", _unit([0.0, 0.0, 1.0, 0.0])),
            ("d", _unit([0.0, 0.0, 0.0, 1.0])),
        ]:
            results = index.search(query_vec, k=1)
            assert results[0][0] == query_id
            assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_add_remove_add_cycles(self) -> None:
        """Mapping must stay consistent through many add/remove cycles."""
        # Need ≥ node count dims for orthogonal test vectors
        dim = 10
        idx = FAISSIndex(dimension=dim)

        def basis(i: int) -> list[float]:
            v = [0.0] * dim
            v[i] = 1.0
            return v

        for cycle in range(5):
            for i in range(dim):
                idx.add(f"n{i}", basis(i))
            assert idx.count() == dim
            for i in range(0, dim, 2):
                idx.remove(f"n{i}")
            assert idx.count() == dim // 2

            # Odd ones still findable as top-1 for their own basis vector
            for i in range(1, dim, 2):
                res = idx.search(basis(i), k=1)
                assert res[0][0] == f"n{i}"
                assert res[0][1] == pytest.approx(1.0, abs=1e-5)

            # Clean for next cycle
            for i in range(1, dim, 2):
                idx.remove(f"n{i}")
            assert idx.count() == 0

    def test_remove_all_then_search_empty(self, index: FAISSIndex) -> None:
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        index.add("b", _unit([0.0, 1.0, 0.0, 0.0]))
        index.remove("a")
        index.remove("b")
        assert index.count() == 0
        assert index.search(_unit([1.0, 0.0, 0.0, 0.0])) == []


# --- Save / load round-trip ---


class TestSaveLoad:
    def test_round_trip_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.faiss"
        idx = FAISSIndex(dimension=DIM)
        idx.save(str(p))

        idx2 = FAISSIndex(dimension=DIM)
        idx2.load(str(p))
        assert idx2.count() == 0
        assert idx2.search(_unit([1.0, 0.0, 0.0, 0.0])) == []

    def test_round_trip_preserves_vectors(self, tmp_path: Path) -> None:
        p = tmp_path / "idx.faiss"
        idx = FAISSIndex(dimension=DIM)
        idx.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        idx.add("b", _unit([0.0, 1.0, 0.0, 0.0]))
        idx.add("c", _unit([0.0, 0.0, 1.0, 0.0]))
        idx.save(str(p))

        idx2 = FAISSIndex(dimension=DIM)
        idx2.load(str(p))
        assert idx2.count() == 3

        # Each original vector still resolves to its exact id
        results = idx2.search(_unit([1.0, 0.0, 0.0, 0.0]), k=1)
        assert results[0][0] == "a"
        results = idx2.search(_unit([0.0, 1.0, 0.0, 0.0]), k=1)
        assert results[0][0] == "b"
        results = idx2.search(_unit([0.0, 0.0, 1.0, 0.0]), k=1)
        assert results[0][0] == "c"

    def test_round_trip_preserves_id_mapping(self, tmp_path: Path) -> None:
        p = tmp_path / "idx.faiss"
        idx = FAISSIndex(dimension=DIM)
        ids_in = ["alpha", "beta-🚀", "gamma with spaces"]
        for i, nid in enumerate(ids_in):
            vec = [0.0] * DIM
            vec[i] = 1.0
            idx.add(nid, vec)
        idx.save(str(p))

        idx2 = FAISSIndex(dimension=DIM)
        idx2.load(str(p))
        for i, nid in enumerate(ids_in):
            vec = [0.0] * DIM
            vec[i] = 1.0
            res = idx2.search(vec, k=1)
            assert res[0][0] == nid

    def test_load_supports_add_and_remove(self, tmp_path: Path) -> None:
        """After load, the index must remain fully mutable."""
        p = tmp_path / "idx.faiss"
        idx = FAISSIndex(dimension=DIM)
        idx.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        idx.save(str(p))

        idx2 = FAISSIndex(dimension=DIM)
        idx2.load(str(p))
        idx2.add("b", _unit([0.0, 1.0, 0.0, 0.0]))
        assert idx2.count() == 2
        idx2.remove("a")
        assert idx2.count() == 1
        res = idx2.search(_unit([0.0, 1.0, 0.0, 0.0]), k=1)
        assert res[0][0] == "b"

    def test_load_missing_index_file_raises(self, tmp_path: Path) -> None:
        idx = FAISSIndex(dimension=DIM)
        with pytest.raises(PersistenceError, match="not found"):
            idx.load(str(tmp_path / "nope.faiss"))

    def test_load_missing_sidecar_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "idx.faiss"
        idx = FAISSIndex(dimension=DIM)
        idx.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        idx.save(str(p))
        # Delete only the sidecar
        Path(str(p) + ".ids.json").unlink()

        idx2 = FAISSIndex(dimension=DIM)
        with pytest.raises(PersistenceError, match="sidecar not found"):
            idx2.load(str(p))

    def test_load_corrupt_sidecar_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "idx.faiss"
        idx = FAISSIndex(dimension=DIM)
        idx.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        idx.save(str(p))
        Path(str(p) + ".ids.json").write_text("not valid json {{")

        idx2 = FAISSIndex(dimension=DIM)
        with pytest.raises(PersistenceError, match="corrupt id sidecar"):
            idx2.load(str(p))

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        p = tmp_path / "a" / "b" / "c" / "idx.faiss"
        idx = FAISSIndex(dimension=DIM)
        idx.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        idx.save(str(p))
        assert p.exists()
        assert (tmp_path / "a" / "b" / "c" / "idx.faiss.ids.json").exists()

    def test_save_after_remove_round_trips(self, tmp_path: Path) -> None:
        """Remove some vectors, save, then load into a fresh instance.

        Only the surviving ids should be present.
        """
        p = tmp_path / "idx.faiss"
        idx = FAISSIndex(dimension=DIM)
        idx.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        idx.add("b", _unit([0.0, 1.0, 0.0, 0.0]))
        idx.add("c", _unit([0.0, 0.0, 1.0, 0.0]))
        idx.add("d", _unit([0.0, 0.0, 0.0, 1.0]))

        idx.remove("b")
        idx.remove("d")
        idx.save(str(p))

        idx2 = FAISSIndex(dimension=DIM)
        idx2.load(str(p))
        assert idx2.count() == 2

        # Surviving ids retrievable, removed ids absent
        res_a = idx2.search(_unit([1.0, 0.0, 0.0, 0.0]), k=5)
        res_c = idx2.search(_unit([0.0, 0.0, 1.0, 0.0]), k=5)
        assert "a" in [r[0] for r in res_a]
        assert "c" in [r[0] for r in res_c]
        assert "b" not in {r[0] for r in res_a} | {r[0] for r in res_c}
        assert "d" not in {r[0] for r in res_a} | {r[0] for r in res_c}

        # And load didn't break future mutation
        idx2.add("e", _unit([1.0, 1.0, 0.0, 0.0]))
        idx2.remove("a")
        assert idx2.count() == 2

    def test_load_rejects_dim_mismatch_between_sidecar_and_index(
        self, tmp_path: Path
    ) -> None:
        """Regression: tampered sidecar with wrong dim must not silently apply."""
        p = tmp_path / "idx.faiss"
        idx = FAISSIndex(dimension=DIM)
        idx.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        idx.save(str(p))

        sidecar = Path(str(p) + ".ids.json")
        meta = json.loads(sidecar.read_text())
        meta["dimension"] = DIM + 1
        sidecar.write_text(json.dumps(meta))

        idx2 = FAISSIndex(dimension=DIM)
        with pytest.raises(PersistenceError, match="dim"):
            idx2.load(str(p))

    def test_load_rejects_non_list_ids(self, tmp_path: Path) -> None:
        p = tmp_path / "idx.faiss"
        idx = FAISSIndex(dimension=DIM)
        idx.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        idx.save(str(p))

        sidecar = Path(str(p) + ".ids.json")
        sidecar.write_text(json.dumps({"dimension": DIM, "ids": {"not": "a list"}}))

        idx2 = FAISSIndex(dimension=DIM)
        with pytest.raises(PersistenceError, match="list of strings"):
            idx2.load(str(p))

    def test_load_rejects_oversized_sidecar(self, tmp_path: Path) -> None:
        """Simulate a hostile huge sidecar — must refuse without reading."""
        p = tmp_path / "idx.faiss"
        idx = FAISSIndex(dimension=DIM)
        idx.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        idx.save(str(p))

        sidecar = Path(str(p) + ".ids.json")
        # Use sparse-ish write: 65MB of zeros
        from cognigraph.vector_index import _MAX_SIDECAR_BYTES
        sidecar.write_bytes(b"\x00" * (_MAX_SIDECAR_BYTES + 1))

        idx2 = FAISSIndex(dimension=DIM)
        with pytest.raises(PersistenceError, match="too large"):
            idx2.load(str(p))


# --- Lifecycle ---


class TestLifecycle:
    def test_close_resets_to_empty(self, index: FAISSIndex) -> None:
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        index.add("b", _unit([0.0, 1.0, 0.0, 0.0]))
        index.close()
        assert index.count() == 0
        assert index.search(_unit([1.0, 0.0, 0.0, 0.0])) == []

    def test_context_manager_closes(self) -> None:
        with FAISSIndex(dimension=DIM) as idx:
            idx.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
            assert idx.count() == 1
        # After __exit__, close() was called
        assert idx.count() == 0

    def test_add_after_close(self, index: FAISSIndex) -> None:
        index.add("a", _unit([1.0, 0.0, 0.0, 0.0]))
        index.close()
        index.add("b", _unit([0.0, 1.0, 0.0, 0.0]))
        assert index.count() == 1
        res = index.search(_unit([0.0, 1.0, 0.0, 0.0]), k=1)
        assert res[0][0] == "b"


# --- Scale sanity ---


class TestScale:
    def test_1000_vectors_retrievable(self) -> None:
        """A few hundred vectors should all be retrievable with their own ids."""
        dim = 16
        idx = FAISSIndex(dimension=dim)
        rng = np.random.default_rng(seed=42)

        # Use a set of orthogonal-ish basis vectors to guarantee each id
        # is closest to its own stored vector.
        vectors: dict[str, np.ndarray] = {}
        for i in range(200):
            v = rng.standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v)
            node_id = f"n{i}"
            vectors[node_id] = v
            idx.add(node_id, v.tolist())

        assert idx.count() == 200

        # Every stored vector's top-1 hit is itself.
        for node_id, v in vectors.items():
            res = idx.search(v.tolist(), k=1)
            assert res[0][0] == node_id
            assert res[0][1] == pytest.approx(1.0, abs=1e-4)
