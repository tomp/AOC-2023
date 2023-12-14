#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 14
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
        O....#....
        O.OO#....#
        .....##...
        OO.#O....O
        .O.....O#.
        O.#..O.#.#
        ..O..#O..O
        .......O..
        #....###..
        #OO..#....
        """,
        136
    ),
]

SAMPLE_CASES2 = [
    (
        """
        O....#....
        O.OO#....#
        .....##...
        OO.#O....O
        .O.....O#.
        O.#..O.#.#
        ..O..#O..O
        .......O..
        #....###..
        #OO..#....
        """,
        64
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

    def neighbors(self) -> "List[Pos]":
        return [
            Pos(self.row, self.col + 1),
            Pos(self.row + 1, self.col),
            Pos(self.row, self.col - 1),
            Pos(self.row - 1, self.col),
        ]

    def north(self) -> "Pos":
        return Pos(self.row - 1, self.col)

    def south(self) -> "Pos":
        return Pos(self.row + 1, self.col)

    def east(self) -> "Pos":
        return Pos(self.row, self.col + 1)

    def west(self) -> "Pos":
        return Pos(self.row, self.col - 1)


EMPTY, CUBE, ROUND = ".", "#", "O"

@dataclass
class Grid:
    grid: dict[Pos, str]

    def __post_init__(self):
        for pos, ch in self.grid.items():
            # print(f">> @{pos} is {ch}")
            assert ch in (CUBE, ROUND)

    def __str__(self):
        lines = []
        rmin, rmax, cmin, cmax = self.bounds()
        for r in range(rmin, rmax+1):
            line = []
            for c in range(cmin, cmax+1):
                line.append(self.at(Pos(r, c)))
            lines.append("".join(line))
        return "\n".join(lines)

    def bounds(self) -> Tuple[int, int, int, int]:
        rownums = [pos.row for pos in self.grid.keys()]
        rmin, rmax = min(rownums), max(rownums)
        colnums = [pos.col for pos in self.grid.keys()]
        cmin, cmax = min(colnums), max(colnums)
        return rmin, rmax, cmin, cmax

    def size(self) -> Tuple[int, int]:
        rmin, rmax, cmin, cmax = self.bounds()
        return rmax - rmin + 1, cmax - cmin + 1

    def at(self, pos: Pos) -> str:
        return self.grid[pos]

    def load(self) -> int:
        result = 0
        _, rmax, _, _ = self.bounds()
        for pos, item in self.grid.items():
            if item == ROUND:
                result += rmax + 1 - pos.row
        return result

    def cycle(self) -> None:
        # print("\nTilt north...")
        self.tilt_north()
        # print(str(self))
        # print("\nTilt west...")
        self.tilt_west()
        # print(str(self))
        # print("\nTilt south...")
        self.tilt_south()
        # print(str(self))
        # print("\nTilt east...")
        self.tilt_east()
        # print(str(self))

    def tilt_north(self) -> None:
        new_grid = defaultdict(lambda: EMPTY)
        rmin, rmax, cmin, cmax = self.bounds()
        for col in range(cmin, cmax+1):
            open_row = None
            for row in range(rmin, rmax+1):
                pos = Pos(row, col)
                item = self.at(pos)
                if item == EMPTY:
                    if open_row is None:
                        open_row = row
                elif item == CUBE:
                    new_grid[pos] = item
                    open_row = None
                else: # item is ROUND
                    if open_row is None:
                        new_grid[pos] = item
                    else:
                        new_grid[Pos(open_row, col)] = item
                        open_row += 1
        self.grid = new_grid

    def tilt_east(self) -> None:
        new_grid = defaultdict(lambda: EMPTY)
        rmin, rmax, cmin, cmax = self.bounds()
        for row in range(rmin, rmax+1):
            open_col = None
            for col in range(cmax, cmin-1, -1):
                pos = Pos(row, col)
                item = self.at(pos)
                if item == EMPTY:
                    if open_col is None:
                        open_col = col
                elif item == CUBE:
                    new_grid[pos] = item
                    open_col = None
                else: # item is ROUND
                    if open_col is None:
                        new_grid[pos] = item
                    else:
                        new_grid[Pos(row, open_col)] = item
                        open_col -= 1
        self.grid = new_grid

    def tilt_west(self) -> None:
        new_grid = defaultdict(lambda: EMPTY)
        rmin, rmax, cmin, cmax = self.bounds()
        for row in range(rmin, rmax+1):
            open_col = None
            for col in range(cmin, cmax+1):
                pos = Pos(row, col)
                item = self.at(pos)
                if item == EMPTY:
                    if open_col is None:
                        open_col = col
                elif item == CUBE:
                    new_grid[pos] = item
                    open_col = None
                else: # item is ROUND
                    if open_col is None:
                        new_grid[pos] = item
                    else:
                        new_grid[Pos(row, open_col)] = item
                        open_col += 1
        self.grid = new_grid

    def tilt_south(self) -> None:
        new_grid = defaultdict(lambda: EMPTY)
        rmin, rmax, cmin, cmax = self.bounds()
        for col in range(cmin, cmax+1):
            open_row = None
            for row in range(rmax, rmin-1, -1):
                pos = Pos(row, col)
                item = self.at(pos)
                if item == EMPTY:
                    if open_row is None:
                        open_row = row
                elif item == CUBE:
                    new_grid[pos] = item
                    open_row = None
                else: # item is ROUND
                    if open_row is None:
                        new_grid[pos] = item
                    else:
                        new_grid[Pos(open_row, col)] = item
                        open_row -= 1
        self.grid = new_grid


def parse_input(lines):
    grid = defaultdict(lambda: EMPTY)
    for row, line in enumerate(lines):
        assert line
        for col, ch in enumerate(line):
            if ch != EMPTY:
                grid[Pos(row, col)] = ch
    return Grid(grid)

def list2str(vals) -> str:
    return repr(vals)

def find_cycle(hist: list[int]) -> Tuple[int, int]:
    npatt = 8
    n = len(hist)
    if n < npatt:
        return 0, 0

    lastmatch = 0
    lastgap = 0
    patt = hist[-npatt:]
    for end in range(n, -1, -1):
        start = end - npatt
        if hist[start:end] == patt:
            if lastmatch:
                gap = lastmatch - end
                if lastgap and gap == lastgap:
                    return (end - gap, gap)
                lastgap = gap
            lastmatch = end

    return 0, 0

def solve2(lines: Lines) -> int:
    """Solve the problem."""
    target = 1000000000
    grid = parse_input(lines)
    hist = []
    while True:
        load = grid.load()
        hist.append(load)
        offset, period = find_cycle(hist)
        if period:
            break
        grid.cycle()

    idx = offset + ((target - offset) % period)

    return hist[idx]

def solve(lines: Lines) -> int:
    """Solve the problem."""
    grid = parse_input(lines)
    print("Initial grid...")
    print(str(grid))
    print("\nTilt north...")
    grid.tilt_north()
    print(str(grid))

    load = grid.load()
    return load


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
    assert result == 105461
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
    assert result == 102829
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    example2()
    part2(input_lines)
