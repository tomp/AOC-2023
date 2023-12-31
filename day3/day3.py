#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 3
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
        467..114..
        ...*......
        ..35..633.
        ......#...
        617*......
        .....+.58.
        ..592.....
        ......755.
        ...$.*....
        .664.598..
        """,
        4361
    ),
]

SAMPLE_CASES2 = [
    (
        """
        467..114..
        ...*......
        ..35..633.
        ......#...
        617*......
        .....+.58.
        ..592.....
        ......755.
        ...$.*....
        .664.598..
        """,
        467835
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

@dataclass(order=True, frozen=True)
class Pos():
    row: int
    col: int

    def __str__(self) -> str:
        return f"({self.row},{self.col})"

    @classmethod
    def parse_xy(cls, text) -> "Pos":
        x, y = text.split(",")
        row, col = int(y.strip()), int(x.strip())
        return cls(row, col)

    def neighbors(self) -> "List[Pos]":
        return [
            Pos(self.row, self.col + 1),
            Pos(self.row + 1, self.col),
            Pos(self.row, self.col - 1),
            Pos(self.row - 1, self.col),
        ]

NUM_RE = re.compile(r"\b(\d+)\b")
SYMBOL_RE = re.compile(r"([^\d.])")

def parse_lines(lines):
    numbers = [] # List of tuples (number, row, start, end)
    symbols = {} # map Pos to symbol

    for row, line in enumerate(lines):
        for m in NUM_RE.finditer(line):
            num = m.group(1)
            item = (int(num), row, m.start(), m.end())
            numbers.append(item)

        for m in SYMBOL_RE.finditer(line):
            pos = Pos(row, m.start())
            symbols[pos] = m.group(1)

    return numbers, symbols

def neighboring_symbols(row, start, end, symbols) -> list[Pos]:
    result = []
    for col in range(start-1, end+1):
        if Pos(row-1, col) in symbols:
            result.append(Pos(row-1, col))
        if Pos(row+1, col) in symbols:
            result.append(Pos(row+1, col))
    if Pos(row, start-1) in symbols:
        result.append(Pos(row, start-1))
    if Pos(row, end) in symbols:
        result.append(Pos(row, end))
    return result


def solve2(lines: Lines) -> int:
    """Solve the problem."""
    result = 0
    numbers, symbols = parse_lines(lines)
    neighbors = defaultdict(list)
    for num, row, start, end in numbers:
        for pos in neighboring_symbols(row, start, end, symbols):
            neighbors[pos].append(num)

    for pos, nums in neighbors.items():
        if len(nums) == 2:
            result += nums[0] * nums[1]

    return result

def solve(lines: Lines) -> int:
    """Solve the problem."""
    result = 0
    numbers, symbols = parse_lines(lines)
    for num, row, start, end in numbers:
        if neighboring_symbols(row, start, end, symbols):
            result += num
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
    assert result == 527446
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
    assert result == 73201705
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    example2()
    part2(input_lines)
