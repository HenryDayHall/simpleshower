"""
The learning rate schedule (warmup, fixed until the losses stop falling, then
a cosine) and which epochs' checkpoints are kept.

    pytest test/test_schedule.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simpleshower.data import DataError  # noqa: E402
from simpleshower.train import (  # noqa: E402
    DEFAULTS,
    Schedule,
    check_schedule_settings,
    keeps_checkpoint,
)


def losses(multiplicity, detector=2.0, pdg=1.6, energy=0.1):
    return {"multiplicity": multiplicity, "detector": detector, "pdg": pdg, "energy": energy}


def test_warmup_then_fixed():
    schedule = Schedule(warmup=10, plateau_epochs=3, cosine_steps=50)
    assert schedule.factor(0) == pytest.approx(0.1)
    assert schedule.factor(9) == pytest.approx(1.0)
    assert schedule.phase(5) == "warmup"
    # however long, until the losses stop falling
    assert schedule.factor(10) == schedule.factor(10_000) == 1.0
    assert schedule.phase(10_000) == "fixed"


def test_anneals_once_no_loss_has_fallen_for_three_epochs():
    schedule = Schedule(warmup=0, plateau_epochs=3, cosine_steps=40)
    step = 0
    history = [
        losses(2.0),  # all new
        losses(2.1),  # none
        losses(2.1, detector=1.9),  # one falls, so it's not a plateau
        losses(2.2),
        losses(2.0, energy=0.2),  # back to its lowest isn't lower
    ]
    for epoch_losses in history:
        step += 10
        assert not schedule.end_epoch(epoch_losses, step)
    assert schedule.epochs_without_fall == 2
    # the third epoch in a row without a new lowest
    step += 10
    assert schedule.end_epoch(losses(2.5), step)
    assert schedule.anneal_from == step
    # starts annealing only once
    assert not schedule.end_epoch(losses(2.5), step + 10)
    assert schedule.factor(step - 1) == 1.0
    assert schedule.factor(step) == pytest.approx(1.0)
    assert schedule.factor(step + 20) == pytest.approx(0.55)  # half way
    assert schedule.factor(step + 40) == pytest.approx(0.1)
    assert schedule.factor(step + 400) == pytest.approx(0.1)
    assert schedule.phase(step + 20) == "annealing"
    assert not schedule.annealed(step + 39)
    assert schedule.annealed(step + 40) and schedule.phase(step + 40) == "annealed"


def test_losses_missing_or_not_counted():
    # with nothing held out but multiplicity, as with no deposits at all
    schedule = Schedule(warmup=0, plateau_epochs=1, cosine_steps=10)
    assert not schedule.end_epoch({"multiplicity": 1.0, "total": 9.0}, 5)
    # the total and the accuracies aren't among the four
    assert schedule.end_epoch({"multiplicity": 1.0, "total": 1.0, "pdg_accuracy": 0.9}, 10)


def test_not_annealing_during_warmup():
    schedule = Schedule(warmup=100, plateau_epochs=1, cosine_steps=10)
    schedule.end_epoch(losses(1.0), 10)
    assert schedule.end_epoch(losses(1.0), 20)
    assert schedule.anneal_from == 100
    assert schedule.factor(50) == pytest.approx(0.51)
    assert schedule.factor(105) == pytest.approx(0.55)


def test_no_cosine():
    schedule = Schedule(warmup=0, plateau_epochs=1, cosine_steps=0)
    schedule.end_epoch(losses(1.0), 10)
    schedule.end_epoch(losses(1.0), 20)
    assert schedule.annealed(20) and schedule.factor(20) == pytest.approx(0.1)


def test_kept_checkpoints():
    kept = [epoch for epoch in range(0, 1001) if keeps_checkpoint(epoch)]
    assert kept == (
        [0, 2, 4, 8, 10]
        + list(range(20, 101, 10))
        + list(range(200, 1001, 100))
    )


def test_schedule_settings():
    check_schedule_settings(DEFAULTS)
    assert DEFAULTS["epochs"] == -1
    check_schedule_settings(dict(DEFAULTS, epochs=7, cosine_epochs=0))
    for wrong in [
        {"epochs": 0},
        {"epochs": -2},
        {"epochs": 2.5},
        {"plateau_epochs": 0},
        {"cosine_epochs": -1},
        {"cosine_epochs": True},
    ]:
        with pytest.raises(DataError):
            check_schedule_settings(dict(DEFAULTS, **wrong))
