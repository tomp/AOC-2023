#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 11
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
        ...#......
        .......#..
        #.........
        ..........
        ......#...
        .#........
        .........#
        ..........
        .......#..
        #...#.....
        """,
        374
    ),
]

SAMPLE_CASES2 = [
    (
        """
        ...#......
        .......#..
        #.........
        ..........
        ......#...
        .#........
        .........#
        ..........
        .......#..
        #...#.....
        """,
        1030
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

# Solution


@dataclass(order=True, frozen=True)
class Pos():
    row: int
    col: int

    def __str__(self) -> str:
        return f"({self.row},{self.col})"

    def dist(self, other: "Pos") -> int:
        return abs(other.row - self.row) + abs(other.col - self.col)


GALAXY, SPACE = "#", "."

def parse_input(lines) -> list[Pos]:
    galaxies = []
    for row, line in enumerate(lines):
        for col, ch in enumerate(line):
            if ch == GALAXY:
                galaxies.append(Pos(row, col))
    return galaxies


def expand_universe(galaxies, factor=2):

    rows, cols = set(), set()
    for pos in galaxies:
        rows.add(pos.row)
        cols.add(pos.col)

    rowmap = {}
    empties = 0
    for row in range(min(rows), max(rows)+1):
        if row not in rows:
            empties += factor - 1
        else:
            rowmap[row] = row + empties

    colmap = {}
    empties = 0
    for col in range(min(cols), max(cols)+1):
        if col not in cols:
            empties += factor - 1
        else:
            colmap[col] = col + empties

    expanded_galaxies = []
    for pos in galaxies:
        newrow, newcol = rowmap[pos.row], colmap[pos.col]
        expanded_galaxies.append(Pos(newrow, newcol))

    return expanded_galaxies


def solve(lines: Lines, factor=2) -> int:
    """Solve the problem."""
    galaxies = parse_input(lines)
    count = len(galaxies)

    galaxies = expand_universe(galaxies, factor)
    result = 0
    for i in range(count-1):
        for j in range(i+1, count):
            dij = galaxies[i].dist(galaxies[j])
            result += dij

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
    assert result == 9769724
    print("= " * 32)


# PART 2

def example2() -> None:
    """Run example for problem with input arguments."""
    print("EXAMPLE 2:")
    for text, expected in SAMPLE_CASES2:
        lines = load_text(text)
        result = solve(lines, 10)
        print(f"'{text}' -> {result} (expected {expected})")
        assert result == expected
    print("= " * 32)

def part2(lines: Lines) -> None:
    print("PART 2:")
    result = solve(lines, 1000000)
    print(f"result is {result}")
    assert result == 603020563700
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    example2()
    part2(input_lines)
