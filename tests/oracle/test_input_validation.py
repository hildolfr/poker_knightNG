import pytest

from poker_knight_ng.reference.cards import ReferenceCardError
from poker_knight_ng.reference.enumerate import ExactOracleError, enumerate_fixed_holes, enumerate_unknown_opponent, evaluate_terminal


class BoundedIterator:
    def __init__(self, values, limit):
        self.values = iter(values)
        self.limit = limit
        self.calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        if self.calls > self.limit:
            raise AssertionError("input was consumed beyond its topology bound")
        return next(self.values)


class FailingIterable:
    def __init__(self, values, failure, phase):
        self.values = values
        self.failure = failure
        self.phase = phase

    def __iter__(self):
        if self.phase == "iter":
            raise self.failure("attacker-controlled iterator text")
        return FailingIterator(self.values, self.failure, self.phase)


class FailingIterator:
    def __init__(self, values, failure, phase):
        self.values = iter(values)
        self.failure = failure
        self.phase = phase

    def __iter__(self):
        return self

    def __next__(self):
        if self.phase == "next":
            raise self.failure("attacker-controlled iterator text")
        return next(self.values)


def test_topology_rejects_bad_lengths_and_cross_player_duplicates():
    with pytest.raises(ExactOracleError):
        evaluate_terminal(["As"], ["2s", "3h", "4d", "5c", "6s"], [["Kh", "Qh"]])
    with pytest.raises(ExactOracleError):
        evaluate_terminal(["As", "Kd"], ["2s", "3h", "4d", "5c", "6s"], [["As", "Qh"]])


def test_card_boundary_and_generator_inputs_follow_reference_policy():
    with pytest.raises(ReferenceCardError):
        enumerate_unknown_opponent([True, "Kd"], ["2s", "3h", "4d"])
    assert enumerate_unknown_opponent((card for card in ["As", "Kd"]), (card for card in ["2s", "3h", "4d", "5c", "6s"])).completed_trials == 990


def test_topology_collection_is_bounded_for_finite_and_infinite_iterators():
    hero = BoundedIterator(["As", "Kd", "Qc"], 3)
    with pytest.raises(ExactOracleError):
        evaluate_terminal(hero, ["2s", "3h", "4d", "5c", "6s"], [["Qh", "Jh"]])
    assert hero.calls == 3

    board = BoundedIterator(["2s", "3h", "4d", "5c", "6s", "7h"], 6)
    with pytest.raises(ExactOracleError):
        evaluate_terminal(["As", "Kd"], board, [["Qh", "Jh"]])
    assert board.calls == 6

    opponents = BoundedIterator([["Qh", "Jh"]] * 7, 7)
    with pytest.raises(ExactOracleError):
        evaluate_terminal(["As", "Kd"], ["2s", "3h", "4d", "5c", "6s"], opponents)
    assert opponents.calls == 7

    hand = BoundedIterator(["Qh", "Jh", "Tc"], 3)
    with pytest.raises(ExactOracleError):
        evaluate_terminal(["As", "Kd"], ["2s", "3h", "4d", "5c", "6s"], [hand])
    assert hand.calls == 3

    with pytest.raises(ExactOracleError):
        enumerate_unknown_opponent(BoundedIterator(iter("As Kd Qc".split()), 3), BoundedIterator(iter("2s 3h 4d 5c 6s 7h".split()), 6))


@pytest.mark.parametrize(
    ("call", "label"),
    [
        (lambda failed: evaluate_terminal(failed, ["2s", "3h", "4d", "5c", "6s"], [["Qh", "Jh"]]), "hero_cards"),
        (lambda failed: evaluate_terminal(["As", "Kd"], failed, [["Qh", "Jh"]]), "board_cards"),
        (lambda failed: evaluate_terminal(["As", "Kd"], ["2s", "3h", "4d", "5c", "6s"], failed), "opponent_holes"),
        (lambda failed: evaluate_terminal(["As", "Kd"], ["2s", "3h", "4d", "5c", "6s"], [failed]), "opponent hand"),
        (lambda failed: enumerate_unknown_opponent(failed, ["2s", "3h", "4d"]), "hero_cards"),
        (lambda failed: enumerate_unknown_opponent(["As", "Kd"], failed), "board_cards"),
        (lambda failed: enumerate_fixed_holes(["As", "Kd"], ["2s", "3h", "4d"], failed), "opponent_holes"),
        (lambda failed: enumerate_fixed_holes(["As", "Kd"], ["2s", "3h", "4d"], [failed]), "opponent hand"),
    ],
)
@pytest.mark.parametrize("failure, phase", [(RuntimeError, "iter"), (ValueError, "next")])
def test_public_oracles_normalize_ordinary_iterator_failures(call, label, failure, phase):
    with pytest.raises(ExactOracleError) as caught:
        call(FailingIterable(["Qh", "Jh"], failure, phase))
    assert str(caught.value) == f"{label} iteration failed"
    assert "attacker-controlled" not in str(caught.value)


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit, GeneratorExit, AssertionError])
def test_iterator_control_flow_and_assertion_failures_propagate(failure):
    with pytest.raises(failure):
        enumerate_unknown_opponent(
            FailingIterable(["As", "Kd"], failure, "next"), ["2s", "3h", "4d"]
        )


def test_malformed_cards_remain_reference_card_errors_after_collection():
    with pytest.raises(ReferenceCardError):
        evaluate_terminal(["As", "not-a-card"], ["2s", "3h", "4d", "5c", "6s"], [["Qh", "Jh"]])


def test_collection_does_not_trust_adversarial_length_hints():
    class LengthHintIterable:
        def __iter__(self):
            return iter(["As", "Kd"])

        def __len__(self):
            raise AssertionError("length hint was trusted")

    assert enumerate_unknown_opponent(
        LengthHintIterable(), ["2s", "3h", "4d", "5c", "6s"]
    ).completed_trials == 990
