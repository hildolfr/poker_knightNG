"""Unit tests for card utilities."""

import pytest
import sys
sys.path.insert(0, '/home/user/Documents/poker_knightNG/src')

from poker_knight_ng.card_utils import (
    card_to_int, int_to_card, parse_hand, hand_to_string,
    get_rank, get_suit, make_card, create_deck_mask, validate_cards
)


class TestCardRepresentation:
    """Test card representation and conversion."""
    
    def test_card_to_int_conversion(self):
        """Test converting card strings to integers."""
        # Test all 52 cards
        expected_values = {
            '2c': 0, '2d': 1, '2h': 2, '2s': 3,
            '3c': 4, '3d': 5, '3h': 6, '3s': 7,
            '4c': 8, '4d': 9, '4h': 10, '4s': 11,
            '5c': 12, '5d': 13, '5h': 14, '5s': 15,
            '6c': 16, '6d': 17, '6h': 18, '6s': 19,
            '7c': 20, '7d': 21, '7h': 22, '7s': 23,
            '8c': 24, '8d': 25, '8h': 26, '8s': 27,
            '9c': 28, '9d': 29, '9h': 30, '9s': 31,
            'Tc': 32, 'Td': 33, 'Th': 34, 'Ts': 35,
            'Jc': 36, 'Jd': 37, 'Jh': 38, 'Js': 39,
            'Qc': 40, 'Qd': 41, 'Qh': 42, 'Qs': 43,
            'Kc': 44, 'Kd': 45, 'Kh': 46, 'Ks': 47,
            'Ac': 48, 'Ad': 49, 'Ah': 50, 'As': 51
        }
        
        for card_str, expected in expected_values.items():
            assert card_to_int(card_str) == expected
            # Test case insensitivity
            assert card_to_int(card_str.upper()) == expected
            assert card_to_int(card_str.lower()) == expected
    
    def test_int_to_card_conversion(self):
        """Test converting integers to card strings."""
        # Test with Unicode symbols
        assert int_to_card(0) == '2♣'
        assert int_to_card(51) == 'A♠'
        assert int_to_card(46) == 'K♥'
        assert int_to_card(41) == 'Q♦'
        
        # Test with letter symbols
        assert int_to_card(0, unicode=False) == '2c'
        assert int_to_card(51, unicode=False) == 'As'
        assert int_to_card(46, unicode=False) == 'Kh'
        assert int_to_card(41, unicode=False) == 'Qd'
    
    def test_round_trip_conversion(self):
        """Test that conversions are reversible."""
        for i in range(52):
            # Unicode round trip
            card_str = int_to_card(i, unicode=True)
            # Convert back using letter format since that's what card_to_int expects
            letter_str = int_to_card(i, unicode=False)
            assert card_to_int(letter_str) == i
            
            # Letter round trip
            card_str = int_to_card(i, unicode=False)
            assert card_to_int(card_str) == i
    
    def test_invalid_card_strings(self):
        """Test error handling for invalid card strings."""
        with pytest.raises(ValueError, match="Invalid card string"):
            card_to_int("A")  # Too short
        
        with pytest.raises(ValueError, match="Invalid card string"):
            card_to_int("Asd")  # Too long
        
        with pytest.raises(ValueError, match="Invalid rank"):
            card_to_int("Xs")  # Invalid rank
        
        with pytest.raises(ValueError, match="Invalid suit"):
            card_to_int("Ax")  # Invalid suit
    
    def test_invalid_card_integers(self):
        """Test error handling for invalid card integers."""
        with pytest.raises(ValueError, match="Invalid card integer"):
            int_to_card(-1)
        
        with pytest.raises(ValueError, match="Invalid card integer"):
            int_to_card(52)


class TestHandParsing:
    """Test hand parsing and string conversion."""
    
    def test_parse_hand(self):
        """Test parsing hand strings."""
        # Two card hand
        assert parse_hand("AsKs") == [51, 47]
        assert parse_hand("AhKd") == [50, 45]
        
        # Five card hand
        assert parse_hand("AsKsQsJsTs") == [51, 47, 43, 39, 35]
        
        # Mixed case
        assert parse_hand("asKS") == [51, 47]
    
    def test_hand_to_string(self):
        """Test converting card lists to strings."""
        # Unicode output
        assert hand_to_string([51, 47]) == "A♠K♠"
        assert hand_to_string([50, 45]) == "A♥K♦"
        
        # Letter output
        assert hand_to_string([51, 47], unicode=False) == "AsKs"
        assert hand_to_string([50, 45], unicode=False) == "AhKd"
    
    def test_parse_hand_errors(self):
        """Test error handling in hand parsing."""
        # Odd length
        with pytest.raises(ValueError, match="must have even length"):
            parse_hand("AsK")
        
        # Duplicate cards
        with pytest.raises(ValueError, match="Duplicate card"):
            parse_hand("AsAs")


class TestRankSuitOperations:
    """Test rank and suit extraction/creation."""
    
    def test_get_rank(self):
        """Test rank extraction."""
        assert get_rank(0) == 0    # 2
        assert get_rank(51) == 12  # A
        assert get_rank(44) == 11  # K
        assert get_rank(40) == 10  # Q
        assert get_rank(36) == 9   # J
        assert get_rank(32) == 8   # T
    
    def test_get_suit(self):
        """Test suit extraction."""
        assert get_suit(0) == 0   # clubs
        assert get_suit(1) == 1   # diamonds
        assert get_suit(2) == 2   # hearts
        assert get_suit(3) == 3   # spades
        assert get_suit(51) == 3  # spades (Ace of spades)
    
    def test_make_card(self):
        """Test card creation from rank and suit."""
        assert make_card(0, 0) == 0    # 2♣
        assert make_card(12, 3) == 51  # A♠
        assert make_card(11, 2) == 46  # K♥
        assert make_card(10, 1) == 41  # Q♦
    
    def test_make_card_errors(self):
        """Test error handling in card creation."""
        with pytest.raises(ValueError, match="Invalid rank"):
            make_card(-1, 0)
        
        with pytest.raises(ValueError, match="Invalid rank"):
            make_card(13, 0)
        
        with pytest.raises(ValueError, match="Invalid suit"):
            make_card(0, -1)
        
        with pytest.raises(ValueError, match="Invalid suit"):
            make_card(0, 4)


class TestDeckOperations:
    """Test deck mask and card validation."""
    
    def test_create_deck_mask(self):
        """Test deck mask creation."""
        import numpy as np
        
        # Full deck
        mask = create_deck_mask()
        assert mask.sum() == 52
        assert len(mask) == 52
        assert mask.all()
        
        # Exclude some cards
        excluded = [0, 51, 25]  # 2♣, A♠, 8♦
        mask = create_deck_mask(excluded)
        assert mask.sum() == 49
        assert not mask[0]
        assert not mask[51]
        assert not mask[25]
        assert mask[1]
    
    def test_validate_cards(self):
        """Test card validation."""
        # Valid cards
        validate_cards([0, 1, 2, 3])  # Should not raise
        validate_cards([51, 50, 49])  # Should not raise
        
        # Invalid card values
        with pytest.raises(ValueError, match="Invalid card integer"):
            validate_cards([0, 52])
        
        with pytest.raises(ValueError, match="Invalid card integer"):
            validate_cards([-1, 0])
        
        # Duplicate cards
        with pytest.raises(ValueError, match="Duplicate card"):
            validate_cards([0, 1, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])