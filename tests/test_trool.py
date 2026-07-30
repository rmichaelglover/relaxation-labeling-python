from triangles_gt_squares import Tri, Trool


def test_subtraction_is_closed_over_trinary_values():
    assert (Trool(1) - Trool(-1)).value == 1
    assert (Trool(-1) - Trool(1)).value == -1
    assert (Trool(1) - Trool(1)).value == 0


def test_example_tri_difference_has_a_finite_norm():
    left = Tri(Trool(1), Trool(0), Trool(-1))
    right = Tri(Trool(-1), Trool(1), Trool(0))

    assert (left - right).norm() == 3 ** 0.5
