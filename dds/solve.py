import ctypes
from typing import Dict, List

from bridgebots import (
    BiddingSuit,
    Card,
    Contract,
    Deal,
    Direction,
    PlayerHand,
    Rank,
    Suit,
    trick_evaluator,
)

from dds import dds


def bidding_suit_to_dds(suit: BiddingSuit) -> int:
    if suit == BiddingSuit.NO_TRUMP:
        return 4
    return 3 - suit.value[0]


def dds_to_suit(suit: int) -> Suit:
    return Suit(3 - suit)


def suit_to_dds(suit: Suit) -> int:
    return 3 - suit.value


def rank_to_dds(rank: Rank) -> int:
    return rank.value[0]


def dds_to_rank(rank: int) -> Rank:
    return next(r for r in Rank if rank in r.value)


def dds_to_ranks(ranks: int) -> List[Rank]:
    return [r for r in reversed(Rank) if (1 << r.value[0]) & ranks]


def direction_to_dds(dir: Direction) -> int:
    return dir.value


def player_suits_to_dds(suits: Dict[Suit, List[Rank]]) -> str:
    return ".".join(
        [
            "".join(list(map(lambda rank: rank.abbreviation(), suits[suit])))
            for suit in reversed(Suit)
        ]
    )


def player_hand_unplayed(
    hand: PlayerHand, cards_played: list[Card]
) -> Dict[Suit, List[Rank]]:
    def suit_cards(suit: Suit):
        return [
            rank for rank in hand.suits[suit] if Card(suit, rank) not in cards_played
        ]

    return dict([(item, suit_cards(item)) for item in reversed(Suit)])


def get_current_trick(
    deal: Deal, contract: Contract, declarer: Direction, play: List[Card]
):
    trump_suit = contract.suit
    tricks = [play[i : i + 4] for i in range(0, len(play), 4)]

    current_trick = []
    if len(tricks) and len(tricks[-1]) < 4:
        current_trick = tricks[-1]

    last_trick = []
    if len(current_trick) and len(tricks) > 1:
        last_trick = tricks[-2]
    elif len(current_trick) == 0 and len(tricks):
        last_trick = tricks[-1]

    leader: Direction = None
    if len(last_trick):
        evaluator = trick_evaluator(trump_suit, last_trick[0].suit)
        _, winning_card = max(enumerate(last_trick), key=lambda c: evaluator(c[1]))
        for player, cards in deal.player_cards.items():
            if winning_card in cards:
                leader = player
                break
    else:
        leader = declarer.next()

    return current_trick, leader


def dds_solve_board(
    deal: Deal, contract: Contract, declarer: Direction, play: List[str]
):
    current_trick, leader = get_current_trick(deal, contract, declarer, play)
    trump = bidding_suit_to_dds(contract.suit)
    futureTricks = dds.futureTricks()
    dealPbn = dds.dealPBN()
    dealPbn.trump = trump
    dealPbn.first = direction_to_dds(leader)
    for i in range(3):
        dealPbn.currentTrickSuit[i] = 0
        dealPbn.currentTrickRank[i] = 0
        if i < len(current_trick):
            dealPbn.currentTrickSuit[i] = suit_to_dds(current_trick[i].suit)
            dealPbn.currentTrickRank[i] = rank_to_dds(current_trick[i].rank)

    hand_strs = " ".join(
        map(
            lambda hand: player_suits_to_dds(player_hand_unplayed(hand, play)),
            [deal.hands[dir] for dir in Direction],
        )
    )
    dealPbn.remainCards = f"N:{hand_strs}".encode("utf-8")

    res = dds.SolveBoardPBN(dealPbn, -1, 3, 0, ctypes.pointer(futureTricks), 0)

    if res < 0:
        raise Exception(f"libdds error: {dds.get_error_message(res)}")

    card_results = {}
    fut = ctypes.pointer(futureTricks)
    for i in range(fut.contents.cards):
        suit = dds_to_suit(fut.contents.suit[i])
        rank = dds_to_rank(fut.contents.rank[i])
        score = fut.contents.score[i]
        card_results[Card(suit, rank)] = score
        for eq in dds_to_ranks(fut.contents.equals[i]):
            card_results[Card(suit, eq)] = score

    return card_results
