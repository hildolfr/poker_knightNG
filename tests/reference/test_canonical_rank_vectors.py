import pytest

from poker_knight_ng.reference.cards import CARD_DECK, Card, ReferenceCardError, parse_cards
from poker_knight_ng.reference.evaluator import best_five, score_five


def test_canonical_card_ids_and_immutable_complete_deck():
    assert Card.parse("2s").card_id == 0
    assert Card.parse("As").card_id == 12
    assert Card.parse("Th").card_id == 21
    assert Card.parse("2c").card_id == 39
    assert len(CARD_DECK) == 52
    assert len(set(CARD_DECK)) == 52
    with pytest.raises(TypeError):
        CARD_DECK[0] = Card.parse("As")


@pytest.mark.parametrize("token", [True, 12, None, "", "A", "Ass", "as", "AS", "A♠", "T h"])
def test_card_parser_rejects_noncanonical_tokens(token):
    with pytest.raises(ReferenceCardError):
        Card.parse(token)


def test_parse_cards_rejects_duplicates():
    with pytest.raises(ReferenceCardError, match="duplicate"):
        parse_cards(["As", "As"])


@pytest.mark.parametrize(
    ("rank", "suit"),
    [
        (True, 0), (2, True), ("2", 0), (2, "s"),
        (1, 0), (15, 0), (-1, 0), (14, -1), (14, 4),
    ],
)
def test_direct_card_construction_rejects_noncanonical_values(rank, suit):
    with pytest.raises(ReferenceCardError):
        Card(rank, suit)


@pytest.mark.parametrize("card", [Card(2, 0), Card(14, 3)])
def test_direct_card_construction_round_trips_canonical_boundaries(card):
    assert Card.parse(card.token) == card
    assert CARD_DECK[card.card_id] == card


@pytest.mark.parametrize("attribute, value", [("rank", True), ("rank", 1), ("rank", 15), ("rank", -1), ("suit", True), ("suit", -1), ("suit", 4)])
def test_evaluator_entry_points_reject_forged_card_values(attribute, value):
    card = Card(14, 0)
    object.__setattr__(card, attribute, value)
    with pytest.raises(ReferenceCardError):
        parse_cards([card])

    from poker_knight_ng.reference.evaluator import best_five, score_five

    hand = [card, Card.parse("Ks"), Card.parse("Qs"), Card.parse("Js"), Card.parse("Ts")]
    with pytest.raises(ReferenceCardError):
        score_five(hand)
    with pytest.raises(ReferenceCardError):
        best_five(hand)


@pytest.mark.parametrize("attribute, value", [("rank", 1), ("rank", 15), ("suit", -1), ("suit", 4)])
def test_card_identity_properties_reject_forged_exact_card_values(attribute, value):
    card = Card(14, 0)
    object.__setattr__(card, attribute, value)
    with pytest.raises(ReferenceCardError):
        _ = card.card_id
    with pytest.raises(ReferenceCardError):
        _ = card.token


class _BenignCardSubclass(Card):
    pass


class _MaskedCard(Card):
    def __getattribute__(self, name):
        if name == "rank":
            return 14
        if name == "suit":
            return 0
        return super().__getattribute__(name)


class _EqualityEvasionCard(Card):
    def __eq__(self, other):
        return False

    def __hash__(self):
        return id(self)


def _subclass_card(card_type, rank=14, suit=0):
    card = object.__new__(card_type)
    object.__setattr__(card, "rank", rank)
    object.__setattr__(card, "suit", suit)
    return card


def _assert_entry_points_reject(card):
    from poker_knight_ng.reference.evaluator import best_five, score_five

    hand = [card, Card.parse("Ks"), Card.parse("Qs"), Card.parse("Js"), Card.parse("Ts")]
    with pytest.raises(ReferenceCardError):
        parse_cards([card])
    with pytest.raises(ReferenceCardError):
        score_five(hand)
    with pytest.raises(ReferenceCardError):
        best_five(hand)


@pytest.mark.parametrize(
    "card",
    [
        _subclass_card(_BenignCardSubclass),
        _subclass_card(_MaskedCard, 99, 99),
        _subclass_card(_EqualityEvasionCard),
    ],
)
def test_evaluator_entry_points_reject_all_card_subclasses(card):
    _assert_entry_points_reject(card)


def test_evaluator_rejects_equality_evasion_subclass_duplicates():
    first = _subclass_card(_EqualityEvasionCard)
    second = _subclass_card(_EqualityEvasionCard)
    assert first != second
    assert len({first, second}) == 2

    from poker_knight_ng.reference.evaluator import best_five, score_five

    with pytest.raises(ReferenceCardError):
        parse_cards([first, second])
    with pytest.raises(ReferenceCardError):
        score_five([first, second, Card.parse("Ks"), Card.parse("Qs"), Card.parse("Js")])
    with pytest.raises(ReferenceCardError):
        best_five([first, second, Card.parse("Ks"), Card.parse("Qs"), Card.parse("Js")])


@pytest.mark.parametrize(
    "card",
    [
        _subclass_card(_BenignCardSubclass),
        _subclass_card(_MaskedCard, 99, 99),
        _subclass_card(_EqualityEvasionCard),
    ],
)
def test_card_identity_properties_reject_all_card_subclasses(card):
    with pytest.raises(ReferenceCardError):
        _ = card.card_id
    with pytest.raises(ReferenceCardError):
        _ = card.token


def test_evaluator_uses_raw_card_fields_after_virtual_access_is_masked(monkeypatch):
    from poker_knight_ng.reference.evaluator import best_five, score_five

    five_cards = tuple(Card.parse(token) for token in ("As", "Kd", "Qh", "Jc", "9s"))
    seven_cards = five_cards + tuple(Card.parse(token) for token in ("8d", "2c"))

    def masked_getattribute(self, name):
        if name in {"rank", "suit", "card_id", "token"}:
            return 2 if name == "rank" else 0 if name == "suit" else "masked"
        return object.__getattribute__(self, name)

    monkeypatch.setattr(Card, "__getattribute__", masked_getattribute)

    assert score_five(five_cards) == (0, 14, 13, 12, 11, 9)
    assert best_five(seven_cards).score == (0, 14, 13, 12, 11, 9)
    assert best_five(seven_cards).cards == ("9s", "As", "Qh", "Kd", "Jc")


class _ImpersonatingToken(str):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return "As"[index]


class _BenignTokenSubclass(str):
    pass


@pytest.mark.parametrize("token", [_ImpersonatingToken("not-a-card"), _BenignTokenSubclass("As")])
def test_reference_entry_points_reject_all_string_subclasses(token):
    hand = [token, "Ks", "Qs", "Js", "Ts"]
    with pytest.raises(ReferenceCardError):
        Card.parse(token)
    with pytest.raises(ReferenceCardError):
        parse_cards([token])
    with pytest.raises(ReferenceCardError):
        score_five(hand)
    with pytest.raises(ReferenceCardError):
        best_five(hand)


def test_reference_entry_points_accept_canonical_exact_strings():
    hand = ["As", "Ks", "Qs", "Js", "Ts"]
    assert Card.parse("As").token == "As"
    assert parse_cards(hand)[0].token == "As"
    assert score_five(hand) == (8, 14, 0, 0, 0, 0)
    assert best_five(hand).cards == ("Ts", "Js", "Qs", "Ks", "As")
