from typing import Dict, List, Tuple

import numpy as np
from bridgebots import BiddingSuit, Card, DealRecord, Direction, trick_evaluator
from bridgebots.lin import parse_lin_str

from dds.solve import dds_solve_board, get_current_trick
from structure import hand_to_input_bin


def card_to_id(card: Card) -> int:
    return card.suit.value * 13 + (card.rank.value[0] - 2)


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
print(f"input_size is {input_size}")


solutions: List[Tuple[DealRecord, int, Dict]] = []
for file_id in range(1, 8):
    file_name = f"{file_id}.lin"
    print(f"file name {file_name}")
    with open(file_name, "r") as f:
        deals = parse_lin_str(f.read())
        print(f"there are {len(deals)} deals")
        for deal in deals:
            board = deal.board_records[0]
            if board.contract.level == 0:
                continue  # skip hands that were passed out
            for i in range(1, 52):
                play_record = board.play_record[:i]
                current_trick, leader = get_current_trick(
                    deal.deal, board.contract, board.declarer, play_record
                )
                current_player = leader.offset(len(current_trick))
                if current_player != board.declarer:  # just declarer for now
                    continue
                solution = dds_solve_board(
                    deal.deal, board.contract, board.declarer, play_record
                )
                if len(solution.keys()) < 2 or next(iter(solution.values())) < 0:
                    continue
                solutions.append((deal, i, solution))

num_samples = len(solutions)
print(f"total rows {num_samples}")

x = np.zeros((num_samples, input_size), np.float16)
y = np.zeros((num_samples, 52), np.float16)

for i, (deal_record, position, solution) in enumerate(solutions):
    x[i] = hand_to_input_bin(deal_record, position)

    # Only assigns '1' for all the 'best' cards, and '0' for everything else.
    # max_score = max([(value, key) for key, value in solution.items()])[0]
    # max_cards = dict(filter(lambda v: v[1] == max_score, solution.items()))
    # y[i] = cards_to_bin(max_cards)

    # An attempt to provide a ranged input based on the expected tricks.
    max_score = max([(value, key) for key, value in solution.items()])[0] + 1
    for card, score in solution.items():
        scaled_score = (score + 1) / max_score
        y[i][card_to_id(card)] = scaled_score


np.save("x.npy", x)
np.save("y.npy", y)
