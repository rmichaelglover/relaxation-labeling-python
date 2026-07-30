import numpy as np
import pytest

from relax import RelaxationLabeling


def test_zero_compatibility_stays_uniform_and_finite():
    compatibility = np.zeros((2, 2, 2, 2))

    result = RelaxationLabeling(compatibility, iterations=2)

    np.testing.assert_allclose(result.support, 0.0)
    np.testing.assert_allclose(result.strength, 0.5)
    assert np.isfinite(result.strength).all()


def test_order_three_vectorized_matches_loop():
    rng = np.random.default_rng(42)
    compatibility = rng.normal(size=(2, 3, 2, 3, 2, 3))

    vectorized = RelaxationLabeling(
        compatibility, UseMatrixMultiplication=True, iterations=2
    )
    loop = RelaxationLabeling(
        compatibility, UseMatrixMultiplication=False, iterations=2
    )

    np.testing.assert_allclose(vectorized.support, loop.support)
    np.testing.assert_allclose(vectorized.strength, loop.strength)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2, 2),
        (2, 2, 3, 2),
        (2, 2, 2, 2, 2, 3),
    ],
)
def test_rejects_invalid_compatibility_shapes(shape):
    with pytest.raises(ValueError, match="compatibility"):
        RelaxationLabeling(np.zeros(shape), iterations=0)


def test_rejects_negative_iterations():
    with pytest.raises(ValueError, match="iterations"):
        RelaxationLabeling(np.zeros((1, 1, 1, 1)), iterations=-1)
