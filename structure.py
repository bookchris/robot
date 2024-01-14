from typing import Dict, List, Tuple

import numpy as np
from bridgebots import (
    BiddingSuit,
    Card,
    DealRecord,
    Direction,
    Rank,
    Suit,
    trick_evaluator,
)

from dds.solve import get_current_trick


def card_to_id(card: Card) -> int:
    return card.suit.value * 13 + (card.rank.value[0] - 2)


def id_to_card(id: int) -> Card:
    suit = Suit(id // 13)
    rank_id = id % 13
    rank = next(r for r in Rank if rank_id in r.value)
    return Card(suit, rank)


def cards_not_played(cards: List[Card], omit: List[str]) -> List[Card]:
    res = []
    for card in cards:
        if card not in omit:
            res.append(card)
    return res


def cards_to_bin(cards: List[Card]):
    res = np.zeros(52)
    for card in cards:
        res[card_to_id(card)] = 1
    return res


def bidding_suit_to_bin(suit: BiddingSuit):
    res = np.zeros(5)
    res[suit.value[0]] = 1
    return res


def played_from_seat(
    play: List[str], trump: BiddingSuit, declarer: Direction, seat: Direction
):
    tricks = [play[i : i + 4] for i in range(0, len(play), 4)]

    leader = declarer.next()

    seat_played = []

    for trick in tricks:
        player = leader
        for card in trick:
            if player == seat:
                seat_played.append(card)
            player = player.next()
        evaluator = trick_evaluator(trump, trick[0].suit)
        winning_index, winning_card = max(
            enumerate(trick), key=lambda c: evaluator(c[1])
        )
        leader = leader.offset(winning_index)

    return seat_played


def shown_out_from_seat(
    play: List[str], trump: BiddingSuit, declarer: Direction, seat: Direction
):
    shown_out = np.zeros(4)
    tricks = [play[i : i + 4] for i in range(0, len(play), 4)]

    leader = declarer.next()

    for trick in tricks:
        player = leader
        for card in trick:
            if player == seat:
                if card.suit != trick[0].suit:
                    shown_out[trick[0].suit.value] = True
            player = player.next()
        evaluator = trick_evaluator(trump, trick[0].suit)
        winning_index, winning_card = max(
            enumerate(trick), key=lambda c: evaluator(c[1])
        )
        leader = leader.offset(winning_index)

    return shown_out


"""
52: declarer hand
52: dummy
52: declarer played
52: dummy played
52: lefty played
52: righty played
52: current trick lefty
52: current trick dummy
52: current trick righty
 5: trump
 4: lefty showed out
 4: righty showed out
"""
input_size = 52 + 52 + 52 + 52 + 52 + 4 + 52 + 4 + 5 + 52 + 52 + 52


def hand_to_input_bin(deal_record: DealRecord, position: int):
    x = np.zeros(input_size)

    board = deal_record.board_records[0]
    play_record = board.play_record[:position]
    deal = deal_record.deal
    contract = board.contract
    trump = contract.suit
    declarer = board.declarer

    current_trick, leader = get_current_trick(deal, contract, declarer, play_record)

    x[0:52] = cards_to_bin(cards_not_played(deal.hands[declarer].cards, play_record))
    x[52:104] = cards_to_bin(
        cards_not_played(deal.hands[declarer.partner()].cards, play_record)
    )
    x[104:156] = cards_to_bin(played_from_seat(play_record, trump, declarer, declarer))
    x[156:208] = cards_to_bin(
        played_from_seat(play_record, trump, declarer, declarer.partner())
    )
    x[208:260] = cards_to_bin(
        played_from_seat(play_record, trump, declarer, declarer.next())
    )
    x[260:312] = cards_to_bin(
        played_from_seat(play_record, trump, declarer, declarer.previous())
    )
    # TODO: Are we structuring the variable current trick in the right way for the network?
    for pos, card in enumerate(current_trick):
        bin = cards_to_bin([card])
        player = leader.offset(pos)
        if player == declarer.next():
            x[312:364] = bin
        elif player == declarer.partner():
            x[364:416] = bin
        elif player == declarer.previous():
            x[416:468] = bin
    x[468:473] = bidding_suit_to_bin(trump)
    x[473:477] = shown_out_from_seat(play_record, trump, declarer, declarer.next())
    x[477:481] = shown_out_from_seat(play_record, trump, declarer, declarer.previous())

    return x
