#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 7
#
from typing import Sequence, Union, Optional, Any, Dict, List, Tuple
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from pprint import pprint
import math
import re

INPUTFILE = "input.txt"

SAMPLE_CASES = [
    (
        """
        32T3K 765
        T55J5 684
        KK677 28
        KTJJT 220
        QQQJA 483
        """,
        6440
    ),
]

SAMPLE_CASES2 = [
    (
        """
        32T3K 765
        T55J5 684
        KK677 28
        KTJJT 220
        QQQJA 483
        """,
        5905
    ),
]



Lines = Sequence[str]
Sections = Sequence[Lines]

# Utility functions

def load_input(infile: str, strip=True, blank_lines=False) -> Lines:
    return load_text(Path(infile).read_text())

def load_text(text: str, strip=True, blank_lines=False) -> Lines:
    if strip:
        lines = [line.strip() for line in text.strip("\n").split("\n")]
    else:
        lines = [line for line in text.strip("\n").split("\n")]
    if blank_lines:
        return lines
    return [line for line in lines if line.strip()]

def parse_sections(lines: Lines) -> Sections:
    result = []
    sect = []
    for line in lines:
        if not line.strip():
            if sect:
                result.append(sect)
            sect = []
        else:
            sect.append(line)
    if sect:
        result.append(sect)
    return result


# Solution

CARDS = "23456789TJQKA"
CARD_VALUE = {card: idx for idx, card in enumerate(CARDS)}

CARDS2 = "J23456789TQKA"
CARD_VALUE2 = {card: idx for idx, card in enumerate(CARDS2)}

TYPE_RANK = {"FIVE": 6, "FOUR": 5, "FULLHOUSE": 4, "THREE": 3, "TWOPAIR": 2, "PAIR": 1, "HIGH": 0}
RANK_NAME = ["HIGH", "PAIR", "TWOPAIR", "THREE", "FULLHOUSE", "FOUR", "FIVE"]


@dataclass()
class Hand:
    cards: str

    def __post_init__(self):
        assert len(self.cards) == 5
        assert all([ch in CARDS for ch in self.cards])
        self.rank = TYPE_RANK[find_type(self.cards)]
        self.key = [self.rank] + [CARD_VALUE[card] for card in self.cards]
        self.rank2 = TYPE_RANK[find_type2(self.cards)]
        self.key2 = [self.rank2] + [CARD_VALUE2[card] for card in self.cards]

    def __str__(self) -> str:
        return f"<Hand({self.cards}) {RANK_NAME[self.rank]}>"


@dataclass()
class Hand2:
    cards: str

    def __post_init__(self):
        assert len(self.cards) == 5
        assert all([ch in CARDS for ch in self.cards])
        self.rank = TYPE_RANK[find_type2(self.cards)]
        self.key = [self.rank] + [CARD_VALUE2[card] for card in self.cards]

    def __str__(self) -> str:
        return f"<Hand({self.cards}) {RANK_NAME[self.rank]}>"

def find_type(hand: str) -> int:
    cards = list(hand)
    cards.sort()

    count = defaultdict(int)
    for card in cards:
        count[card] += 1

    three = ""
    two = ""
    for card, num in count.items():
        if num == 5:
            return "FIVE"
        if num == 4:
            return "FOUR"
        if num == 3:
            three = card
        elif num == 2:
            if two:
                return "TWOPAIR"
            else:
                two = card
    if three and two:
        return "FULLHOUSE"
    if three:
        return "THREE"
    if two:
        return "PAIR"
    return "HIGH"

def find_type2(hand: str) -> int:
    cards = list(hand)
    cards.sort()

    count = defaultdict(int)
    for card in cards:
        count[card] += 1

    jokers = count["J"]
    if jokers == 5:
        return "FIVE"

    three = ""
    two = ""
    for card, num in count.items():
        if num == 0 or card == "J":
            continue
        if num == 5:
            return "FIVE"
        if num == 4:
            if jokers:
                return "FIVE"
            else:
                return "FOUR"
        if num == 3:
            if jokers == 2:
                return "FIVE"
            elif jokers == 1:
                return "FOUR"
            three = card
        elif num == 2:
            if two:
                if jokers:
                    return "FULLHOUSE"
                else:
                    return "TWOPAIR"
            else:
                two = card

    if three and two:
        return "FULLHOUSE"

    if three:
        return "THREE"

    if two:
        if jokers == 3:
            return "FIVE"
        elif jokers == 2:
            return "FOUR"
        elif jokers == 1:
            return "THREE"
        else:
            return "PAIR"

    if jokers == 4:
        return "FIVE"
    elif jokers == 3:
        return "FOUR"
    elif jokers == 2:
        return "THREE"
    elif jokers == 1:
        return "PAIR"

    return "HIGH"


def parse_input(lines) -> list[Hand]:
    hands = []
    for line in lines:
        if not lines:
            continue
        cards, bet = line.strip().split()
        hands.append((Hand(cards), int(bet)))
    return hands

def parse_input2(lines) -> list[Hand]:
    hands = []
    for line in lines:
        if not lines:
            continue
        cards, bet = line.strip().split()
        hands.append((Hand2(cards), int(bet)))
    return hands

def solve2(lines: Lines) -> int:
    """Solve the problem."""
    result = 0

    hands = parse_input2(lines)
    hands.sort(key=lambda v: v[0].key)

    for rank, (hand, bet) in enumerate(hands):
        result += bet * (rank + 1)

    return result

def solve(lines: Lines) -> int:
    """Solve the problem."""
    result = 0

    hands = parse_input(lines)
    hands.sort(key=lambda v: v[0].key)

    for rank, (hand, bet) in enumerate(hands):
        result += bet * (rank + 1)

    return result


# PART 1

def example1() -> None:
    """Run example for problem with input arguments."""
    print("EXAMPLE 1:")
    for text, expected in SAMPLE_CASES:
        lines = load_text(text)
        result = solve(lines)
        print(f"'{text}' -> {result} (expected {expected})")
        assert result == expected
    print("= " * 32)

def part1(lines: Lines) -> None:
    print("PART 1:")
    result = solve(lines)
    print(f"result is {result}")
    assert result == 251806792
    print("= " * 32)


# PART 2

def example2() -> None:
    """Run example for problem with input arguments."""
    print("EXAMPLE 2:")
    for text, expected in SAMPLE_CASES2:
        lines = load_text(text)
        result = solve2(lines)
        print(f"'{text}' -> {result} (expected {expected})")
        assert result == expected
    print("= " * 32)

def part2(lines: Lines) -> None:
    print("PART 2:")
    result = solve2(lines)
    print(f"result is {result}")
    assert result == 252113488
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    example2()
    part2(input_lines)
