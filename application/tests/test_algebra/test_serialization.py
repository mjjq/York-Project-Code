from dataclasses import asdict

from algebraic.mercier_criterion.expand_mercier import (
    RZ_lutjens, calc_derivatives, calc_metric_tensor, calc_jacobian,
    grad_shafranov
)
from algebraic.mercier_criterion.datatypes import State

def test_serialization():
    coords, shaping = RZ_lutjens(False)

    derivatives = calc_derivatives(coords)
    metric = calc_metric_tensor(derivatives)
    j = calc_jacobian(coords, derivatives)

    gs = grad_shafranov(coords, metric, j)

    state = State(
        coords, shaping, derivatives, metric, gs, j
    )

    fname = "test_serialization_state.pkl"
    state.to_disk(fname)

    # Assertions performed in from_disk
    new_state = State.from_disk(fname)

    # Deep comparison of objects
    assert asdict(state)==asdict(new_state)