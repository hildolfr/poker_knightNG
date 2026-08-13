"""Fail-closed v1 wire models and invariant validation."""
from dataclasses import dataclass
import re
from typing import Any, Mapping
from .canonical import canonical_case_hash
from .errors import problem

CARDS = re.compile(r"^[2-9TJQKA][shdc]$")
SEED = re.compile(r"^0x[0-9a-f]{16}$")
U64 = re.compile(r"^(?:0|[1-9][0-9]*)$")
PROVENANCE_128 = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@=-]{0,127}$")
PROVENANCE_256 = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@=-]{0,255}$")
MAX_TRIALS = (2**64 - 1) // 420
RNG_ID = "poker-knight-ng/philox4x32-10"
CATEGORIES = ("high_card", "one_pair", "two_pair", "three_of_a_kind", "straight", "flush", "full_house", "four_of_a_kind", "straight_flush")


def _mapping(value: Any, allowed: set[str], required: set[str], code: str = "UNSUPPORTED_FIELD") -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value) or set(value) - allowed or required - set(value):
        raise problem(code)
    return value


def _u64(value: Any, code: str = "INTERNAL_ERROR") -> int:
    if not isinstance(value, str) or len(value) > 20 or not U64.fullmatch(value) or int(value) > 2**64 - 1:
        raise problem(code)
    return int(value)


def _request_u64(value: Any) -> int:
    n = _u64(value, "INVALID_TRIAL_COUNT")
    if not n or n > MAX_TRIALS:
        raise problem("INVALID_TRIAL_COUNT")
    return n


def _internal_u64(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**64 - 1:
        raise problem(code)
    return value


def _cards(value: Any, *, count: int | None, board: bool = False) -> tuple[str, ...]:
    if not isinstance(value, tuple) or (count is not None and len(value) != count) or (board and len(value) not in (0, 3, 4, 5)):
        raise problem("INVALID_BOARD_LENGTH" if board else "INVALID_CARD")
    if not all(isinstance(card, str) and CARDS.fullmatch(card) for card in value):
        raise problem("INVALID_CARD")
    return value


@dataclass(frozen=True)
class EquityRequest:
    hero_cards: tuple[str, str]
    board_cards: tuple[str, ...]
    opponent_count: int
    requested_trials: int
    seed: int
    backend: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Re-run every normalized invariant before a request is trusted."""
        hero = _cards(self.hero_cards, count=2)
        board = _cards(self.board_cards, count=None, board=True)
        if len(set(hero + board)) != len(hero) + len(board):
            raise problem("DUPLICATE_CARD")
        if isinstance(self.opponent_count, bool) or not isinstance(self.opponent_count, int) or self.opponent_count not in range(1, 7):
            raise problem("INVALID_OPPONENT_COUNT")
        trials = _internal_u64(self.requested_trials, "INVALID_TRIAL_COUNT")
        if not trials or trials > MAX_TRIALS:
            raise problem("INVALID_TRIAL_COUNT")
        _internal_u64(self.seed, "INVALID_SEED")
        if not isinstance(self.backend, str) or self.backend not in ("cpu_reference", "cuda"):
            raise problem("UNSUPPORTED_REQUEST")

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "EquityRequest":
        raw = _mapping(raw, {"contract_version", "hero_cards", "board_cards", "opponent_count", "requested_trials", "seed", "backend", "rng"}, {"contract_version", "hero_cards", "board_cards", "opponent_count", "requested_trials", "seed", "backend", "rng"})
        if raw["contract_version"] != "v1": raise problem("INVALID_CONTRACT_VERSION")
        hero, board = raw["hero_cards"], raw["board_cards"]
        if not isinstance(hero, list) or len(hero) != 2 or not all(isinstance(c, str) and CARDS.fullmatch(c) for c in hero): raise problem("INVALID_CARD")
        if not isinstance(board, list) or len(board) not in (0, 3, 4, 5): raise problem("INVALID_BOARD_LENGTH")
        if not all(isinstance(c, str) and CARDS.fullmatch(c) for c in board): raise problem("INVALID_CARD")
        if len(set(hero + board)) != len(hero) + len(board): raise problem("DUPLICATE_CARD")
        if not isinstance(raw["opponent_count"], str) or raw["opponent_count"] not in {str(x) for x in range(1, 7)}: raise problem("INVALID_OPPONENT_COUNT")
        if not isinstance(raw["seed"], str) or not SEED.fullmatch(raw["seed"]): raise problem("INVALID_SEED")
        if not isinstance(raw["backend"], str) or raw["backend"] not in ("cpu_reference", "cuda"): raise problem("UNSUPPORTED_REQUEST")
        rng = _mapping(raw["rng"], {"algorithm_id", "algorithm_version"}, {"algorithm_id", "algorithm_version"}, "UNSUPPORTED_RNG")
        if rng["algorithm_id"] != RNG_ID or rng["algorithm_version"] != "1": raise problem("UNSUPPORTED_RNG")
        return cls(tuple(hero), tuple(board), int(raw["opponent_count"]), _request_u64(raw["requested_trials"]), int(raw["seed"], 16), raw["backend"])

    def require_available_backend(self) -> None:
        self.validate()
        if self.backend == "cuda": raise problem("BACKEND_UNAVAILABLE")


@dataclass(frozen=True, init=False)
class EquityResult:
    contract_version: str
    backend: str
    rng: tuple[str, str]
    case_hash: str
    seed: int
    requested_trials: int
    completed_trials: int
    unique_wins: int
    ties: int
    tie_by_other_winners: tuple[int, int, int, int, int, int]
    losses: int
    equity_share_units: int
    hero_category_counts: tuple[tuple[str, int], ...]
    probabilities: tuple[tuple[str, int, int], ...]
    timing: int
    provenance: tuple[str, str, str | None, str | None]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("EquityResult instances are created only by EquityResult.parse")

    @classmethod
    def parse(cls, raw: Mapping[str, Any], *, request: EquityRequest) -> "EquityResult":
        if not isinstance(request, EquityRequest): raise problem("INTERNAL_ERROR")
        request.validate()
        required = {"contract_version", "backend", "rng", "case_hash", "seed", "requested_trials", "completed_trials", "unique_wins", "ties", "tie_by_other_winners", "losses", "equity_share_units", "hero_category_counts", "probabilities", "timing", "provenance"}
        raw = _mapping(raw, required, required, "INTERNAL_ERROR")
        if raw["contract_version"] != "v1" or raw["backend"] != request.backend: raise problem("INTERNAL_ERROR")
        rng = _mapping(raw["rng"], {"algorithm_id", "algorithm_version"}, {"algorithm_id", "algorithm_version"}, "INTERNAL_ERROR")
        if rng["algorithm_id"] != RNG_ID or rng["algorithm_version"] != "1": raise problem("INTERNAL_ERROR")
        if raw["case_hash"] != canonical_case_hash(request) or raw["seed"] != f"0x{request.seed:016x}" or raw["requested_trials"] != str(request.requested_trials): raise problem("INTERNAL_ERROR")
        requested, completed = _request_u64(raw["requested_trials"]), _u64(raw["completed_trials"])
        if completed != requested: raise problem("INTERNAL_ERROR")
        wins, ties, losses = (_u64(raw[x]) for x in ("unique_wins", "ties", "losses"))
        bins = _mapping(raw["tie_by_other_winners"], {str(x) for x in range(1, 7)}, {str(x) for x in range(1, 7)}, "INTERNAL_ERROR")
        b = tuple(_u64(bins[str(x)]) for x in range(1, 7))
        if ties != sum(b) or any(b[x] for x in range(request.opponent_count, 6)) or wins + ties + losses != completed: raise problem("INTERNAL_ERROR")
        units = _u64(raw["equity_share_units"])
        if units != 420 * wins + sum(420 // (x + 2) * b[x] for x in range(6)): raise problem("INTERNAL_ERROR")
        cats = _mapping(raw["hero_category_counts"], set(CATEGORIES), set(CATEGORIES), "INTERNAL_ERROR")
        category_counts = tuple((name, _u64(cats[name])) for name in CATEGORIES)
        if sum(value for _, value in category_counts) != completed: raise problem("INTERNAL_ERROR")
        probs = _mapping(raw["probabilities"], {"unique_win", "tie", "loss", "showdown_equity"}, {"unique_win", "tie", "loss", "showdown_equity"}, "INTERNAL_ERROR")
        probability_rows = []
        for name, num, den in (("unique_win", wins, completed), ("tie", ties, completed), ("loss", losses, completed), ("showdown_equity", units, 420 * completed)):
            fraction = _mapping(probs[name], {"numerator", "denominator"}, {"numerator", "denominator"}, "INTERNAL_ERROR")
            if _u64(fraction["numerator"]) != num or _u64(fraction["denominator"]) != den: raise problem("INTERNAL_ERROR")
            probability_rows.append((name, num, den))
        timing = _mapping(raw["timing"], {"total_duration_ns"}, {"total_duration_ns"}, "INTERNAL_ERROR")
        duration = _u64(timing["total_duration_ns"])
        provenance = _mapping(raw["provenance"], {"engine_build_id", "backend_qualification", "device_id", "kernel_id"}, {"engine_build_id", "backend_qualification", "device_id", "kernel_id"}, "INTERNAL_ERROR")
        build, qualification, device, kernel = (provenance[x] for x in ("engine_build_id", "backend_qualification", "device_id", "kernel_id"))
        if not isinstance(build, str) or not PROVENANCE_128.fullmatch(build) or not isinstance(qualification, str) or not PROVENANCE_128.fullmatch(qualification): raise problem("INTERNAL_ERROR")
        if request.backend == "cpu_reference":
            if device is not None or kernel is not None: raise problem("INTERNAL_ERROR")
        elif not isinstance(device, str) or not PROVENANCE_256.fullmatch(device) or not isinstance(kernel, str) or not PROVENANCE_128.fullmatch(kernel): raise problem("INTERNAL_ERROR")
        result = object.__new__(cls)
        for name, value in (("contract_version", "v1"), ("backend", request.backend), ("rng", (RNG_ID, "1")), ("case_hash", raw["case_hash"]), ("seed", request.seed), ("requested_trials", requested), ("completed_trials", completed), ("unique_wins", wins), ("ties", ties), ("tie_by_other_winners", b), ("losses", losses), ("equity_share_units", units), ("hero_category_counts", category_counts), ("probabilities", tuple(probability_rows)), ("timing", duration), ("provenance", (build, qualification, device, kernel))):
            object.__setattr__(result, name, value)
        return result
