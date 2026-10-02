import pytest

from app.kb.vector_codec import pack_embedding, unpack_embedding


def test_round_trip_preserves_values_within_float32_precision():
    vector = [0.1, -0.5, 3.14159, 0.0, 1.0]
    packed = pack_embedding(vector)
    restored = unpack_embedding(packed, dimension=len(vector))
    for original, got in zip(vector, restored):
        assert abs(original - got) < 1e-6


def test_pack_rejects_empty_vector():
    with pytest.raises(ValueError):
        pack_embedding([])


def test_unpack_rejects_dimension_mismatch():
    packed = pack_embedding([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        unpack_embedding(packed, dimension=4)
