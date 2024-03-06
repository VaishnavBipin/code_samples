# Used to test the integrity of the poker infrastructure

import numpy as np
import pytest

from card import Card, Deck, Suit, Hand
class TestPoker:

    @staticmethod
    def compare_cards(list1,list2):
        if list1 is None or list2 is None:
            raise ValueError("list1 or list2 is None")
        for one,two in zip(list1,list2):
            if one.suit != two.suit or one.value != two.value:
                return False
        return True

    def test_deck_construction(self):
        testDeck = Deck()
        assert len(testDeck.deck) == 52

        club = 0
        diamond = 0
        heart = 0
        spade = 0
        valueDist =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        for i in testDeck.deck:
            if i.suit == Suit.Club:
                club += 1
            if i.suit == Suit.Diamond:
                diamond += 1
            if i.suit == Suit.Heart:
                heart += 1
            if i.suit == Suit.Spade:
                spade += 1
            valueDist[i.value] += 1
        assert club==diamond and diamond ==heart and heart ==spade

        assert valueDist[0] == 0
        assert valueDist[1] == 0 
        for i in range(2,15):
            assert valueDist[i] == 4

    # Deck Class Tests
    @pytest.mark.parametrize("num_cards",[
        1,
        2,
        12,
        0,
    ])
    def test_deck_deal(self,num_cards):
        testDeck = Deck()
        draw = testDeck.deal(num_cards)

        assert len(testDeck.deck) == 52-num_cards
        assert len(testDeck.out) == num_cards
    

    # Hand Class Tests
    def test_hand_straight_flush(self):
        # Case 1: normal straight_flush
        normal = [Card(Suit.Club,6),Card(Suit.Club,5),Card(Suit.Club,4),Card(Suit.Club,3),Card(Suit.Club,2)]
        result = Hand.is_straight_flush(normal)
        assert TestPoker.compare_cards(normal,result)

        # Case 2: flush only
        flush = [Card(Suit.Club,7),Card(Suit.Club,5),Card(Suit.Club,4),Card(Suit.Club,4),Card(Suit.Club,2)]
        result = Hand.is_straight_flush(flush)
        assert result is None

        # Case 3: straight only
        straight = [Card(Suit.Diamond,6),Card(Suit.Club,5),Card(Suit.Club,4),Card(Suit.Club,3),Card(Suit.Club,2)]
        result = Hand.is_straight_flush(straight)
        assert result is None

    def test_hand_four_oak(self):
        # Case 1: normal four of a kind
        normal = [Card(Suit.Club,6),Card(Suit.Diamond,6),Card(Suit.Spade,6),Card(Suit.Heart,6),Card(Suit.Club,2)]
        result = Hand.is_four_oak(normal)
        assert TestPoker.compare_cards(normal,result)

        # Case 2: High card in front of the 4oak: need to be reordered
        out_of_order = [Card(Suit.Club,14),
                        Card(Suit.Club,6),
                        Card(Suit.Diamond,6),
                        Card(Suit.Spade,6),
                        Card(Suit.Heart,6)]
        expected = [
            Card(Suit.Club,6),
            Card(Suit.Diamond,6),
            Card(Suit.Spade,6),
            Card(Suit.Heart,6),
            Card(Suit.Club,14)
        ]
        result = Hand.is_four_oak(out_of_order)
        assert TestPoker.compare_cards(expected,result)
        # assert result != out_of_order

        # Case 3: Not 4oak
        wrong = [Card(Suit.Club,6),Card(Suit.Club,5),Card(Suit.Club,4),Card(Suit.Club,3),Card(Suit.Club,2)]
        result = Hand.is_four_oak(wrong)
        assert result is None

    def test_hand_full_house(self):
        # Case 1: Normal
        normal = [
            Card(Suit.Club,6),
            Card(Suit.Diamond,6),
            Card(Suit.Heart,6),
            Card(Suit.Club,3),
            Card(Suit.Diamond,3)
        ]
        expected =[
            Card(Suit.Heart,6),
            Card(Suit.Diamond,6),
            Card(Suit.Club,6),
            Card(Suit.Club,3),
            Card(Suit.Diamond,3)
        ]
        result = Hand.is_full_house(normal)
        assert TestPoker.compare_cards(expected,result)

        # Case 2: Small value full of larger value
        normal_2 = [
                    Card(Suit.Club,6),
                    Card(Suit.Diamond,6),
                    Card(Suit.Heart,3),
                    Card(Suit.Club,3),
                    Card(Suit.Diamond,3)
                ]
        result_2 = [
            Card(Suit.Diamond,3),
            Card(Suit.Club,3),
            Card(Suit.Heart,3),
            Card(Suit.Club,6),
            Card(Suit.Diamond,6)
        ]
        result = Hand.is_full_house(normal_2)
        assert TestPoker.compare_cards(result_2,result)

        # Case 3: Non full house
        wrong = [
            Card(Suit.Club,6),
            Card(Suit.Club,5),
            Card(Suit.Club,4),
            Card(Suit.Club,3),
            Card(Suit.Club,2)
        ]
        result = Hand.is_full_house(wrong)
        assert result is None
    

    def test_hand_is_flush(self):
        # Case 1: flush
        flush = [
            Card(Suit.Club,7),
            Card(Suit.Club,5),
            Card(Suit.Club,3),
            Card(Suit.Club,3),
            Card(Suit.Club,2)
        ]
        result = Hand.is_flush(flush)
        assert TestPoker.compare_cards(flush,result)

        #Case 2: straight flush
        sf = [
            Card(Suit.Club,6),
            Card(Suit.Club,5),
            Card(Suit.Club,4),
            Card(Suit.Club,3),
            Card(Suit.Club,2)
        ]
        result = Hand.is_flush(sf)
        assert TestPoker.compare_cards(sf,result)

        # Case 3 not flush
        wrong = [
            Card(Suit.Diamond,6),
            Card(Suit.Club,5),
            Card(Suit.Club,4),
            Card(Suit.Club,3),
            Card(Suit.Club,2)
            ]
        result = Hand.is_flush(wrong)
        assert result is None
    
    def test_hand_is_straight(self):
        # Case 1: straight
        normal = [
            Card(Suit.Diamond,6),
            Card(Suit.Club,5),
            Card(Suit.Club,4),
            Card(Suit.Club,3),
            Card(Suit.Club,2)
            ]
        result = Hand.is_straight(normal)
        assert TestPoker.compare_cards(normal,result)

        # Case 2: Straight Flush
        sf = [
            Card(Suit.Club,6),
            Card(Suit.Club,5),
            Card(Suit.Club,4),
            Card(Suit.Club,3),
            Card(Suit.Club,2)
        ]
        result = Hand.is_straight(sf)
        assert TestPoker.compare_cards(sf,result)

        # Case 3: not straight
        wrong = [
            Card(Suit.Club,7),
            Card(Suit.Club,5),
            Card(Suit.Club,4),
            Card(Suit.Club,3),
            Card(Suit.Club,2)
        ]
        result = Hand.is_straight(wrong)
        assert result is None
    
    def test_hand_three_oak(self):
        # Case 1: Three oak
        normal =[
            Card(Suit.Club,7),
            Card(Suit.Diamond,7),
            Card(Suit.Heart,7),
            Card(Suit.Club,3),
            Card(Suit.Club,2)
        ]
        expected = [
            Card(Suit.Heart,7),
            Card(Suit.Diamond,7),
            Card(Suit.Club,7),
            Card(Suit.Club,3),
            Card(Suit.Club,2)
        ]
        result = Hand.is_three_oak(normal)
        assert TestPoker.compare_cards(expected,result)
    
 
        # Case 2: Out of order
        oorder =[
            Card(Suit.Spade,9),
            Card(Suit.Club,7),
            Card(Suit.Diamond,7),
            Card(Suit.Heart,7),
            Card(Suit.Club,2)
        ]
        expected = [
            Card(Suit.Heart,7),
            Card(Suit.Diamond,7),
            Card(Suit.Club,7),
            Card(Suit.Spade,9),
            Card(Suit.Club,2)
        ]
        result = Hand.is_three_oak(oorder)
        assert TestPoker.compare_cards(expected,result)

        # Case 4: none (pair)
        wrong =[
            Card(Suit.Spade,9),
            Card(Suit.Club,5),
            Card(Suit.Diamond,5),
            Card(Suit.Heart,2),
            Card(Suit.Club,2)
        ]
        result = Hand.is_three_oak(wrong)
        assert result is None

    def test_hand_two_pair(self):
        # Case 1: normal
        normal = [
            Card(Suit.Club,5),
            Card(Suit.Diamond,5),
            Card(Suit.Heart,3),
            Card(Suit.Club,3),
            Card(Suit.Spade,2)
        ]
        
        expected = [
            Card(Suit.Diamond,5),
            Card(Suit.Club,5),
            Card(Suit.Club,3),
            Card(Suit.Heart,3),
            Card(Suit.Spade,2)
        ]
        result = Hand.is_two_pair(normal)
        assert TestPoker.compare_cards(expected,result)

        # Case 2: out of order 
        oorder = [
            Card(Suit.Spade,8),
            Card(Suit.Club,5),
            Card(Suit.Diamond,5),
            Card(Suit.Heart,3),
            Card(Suit.Club,3)            
        ]

        expected = [
            Card(Suit.Diamond,5),
            Card(Suit.Club,5),
            Card(Suit.Club,3),  
            Card(Suit.Heart,3),
            Card(Suit.Spade,8)
        ]
        result = Hand.is_two_pair(oorder)
        assert TestPoker.compare_cards(expected,result)

        # Case 3: nothing
        wrong = [
            Card(Suit.Spade,8),
            Card(Suit.Club,5),
            Card(Suit.Diamond,4),
            Card(Suit.Heart,3),
            Card(Suit.Club,2)            
        ]
        result = Hand.is_two_pair(wrong)
        assert result is None

    def test_hand_is_pair(self):
        # Case 1: one pair
        normal  = [
            Card(Suit.Spade,8),
            Card(Suit.Club,8),
            Card(Suit.Diamond,4),
            Card(Suit.Heart,3),
            Card(Suit.Club,2)            
        ]
        expected = [
            Card(Suit.Club,8),
            Card(Suit.Spade,8),
            Card(Suit.Diamond,4),
            Card(Suit.Heart,3),
            Card(Suit.Club,2) 

        ]
        result = Hand.is_pair(normal)
        assert TestPoker.compare_cards(expected,result)

        # Case 2: out of order
        oorder  = [
            Card(Suit.Spade,8),
            Card(Suit.Club,7),
            Card(Suit.Diamond,4),
            Card(Suit.Heart,4),
            Card(Suit.Club,2)            
        ]

        expected  = [
            Card(Suit.Heart,4),
            Card(Suit.Diamond,4),  
            Card(Suit.Spade,8),
            Card(Suit.Club,7),
            Card(Suit.Club,2)            
        ]
        result = Hand.is_pair(oorder)
        assert TestPoker.compare_cards(expected,result)

        # Case 3: Multiple pairs
        m_pairs = [
            Card(Suit.Club,9),
            Card(Suit.Spade,8),
            Card(Suit.Club,8),
            Card(Suit.Diamond,4),
            Card(Suit.Heart,4)
        ]
        # only gets the best pair when evaluating; this would get two pairs anyways
        expected = [
            Card(Suit.Club,8),
            Card(Suit.Spade,8),
            Card(Suit.Club,9),
            Card(Suit.Diamond,4),
            Card(Suit.Heart,4)
        ]
        result = Hand.is_pair(m_pairs)
        assert TestPoker.compare_cards(expected,result)

        # Case 4: None
        wrong = [
            Card(Suit.Club,9),
            Card(Suit.Spade,8),
            Card(Suit.Club,7),
            Card(Suit.Diamond,4),
            Card(Suit.Heart,3)
        ]
        result = Hand.is_pair(wrong)
        assert result is None
    
