#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 13
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
        #.##..##.
        ..#.##.#.
        ##......#
        ##......#
        ..#.##.#.
        ..##..##.
        #.#.##.#.

        .##..##.#
        .#.##.#..
        #......##
        #......##
        .#.##.#..
        .##..##..
        .#.##.#.#

        .####.#..
        .#..#..#.
        #....##..
        #....##..
        .#..#...#
        .####....
        .#..#.###
        """,
        12
    ),
    (
        """
        #.##..##.
        ..#.##.#.
        ##......#
        ##......#
        ..#.##.#.
        ..##..##.
        #.#.##.#.

        #...##..#
        #....#..#
        ..##..###
        #####.##.
        #####.##.
        ..##..###
        #....#..#
        """,
        405
    ),
]

SAMPLE_CASES2 = [
    (
        """
        #.##..##.
        ..#.##.#.
        ##......#
        ##......#
        ..#.##.#.
        ..##..##.
        #.#.##.#.

        #...##..#
        #....#..#
        ..##..###
        #####.##.
        #####.##.
        ..##..###
        #....#..#
        """,
        400
    ),
]



Lines = Sequence[str]
Sections = Sequence[Lines]

# Utility functions

def load_input(infile: str, strip=True, blank_lines=False) -> Lines:
    return load_text(Path(infile).read_text(), blank_lines=blank_lines)

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

ASH, ROCK = ".", "#"

@dataclass
class Grid:
    grid: list[list[str]]

    def __post_init__(self):
        self.nrow = len(self.grid)
        self.ncol = len(self.grid[0])
        assert all([len(row) == self.ncol for row in self.grid])

        self.row = ["".join(row) for row in self.grid]
        self.col = []
        for col in range(self.ncol):
            self.col.append("".join([row[col] for row in self.grid]))

        self.row_corr = []
        self.col_corr = []

    def __str__(self):
        lines = []
        for row in self.grid:
            lines.append("".join(row))
        return "\n".join(lines)

    def fix_smudge(self, rsmudge, csmudge) -> "Grid":
        new_grid = []
        for row in self.grid:
            new_grid.append(list(row))

        item = new_grid[rsmudge][csmudge]
        if item == ROCK:
            new_grid[rsmudge][csmudge] = ASH
        else:
            new_grid[rsmudge][csmudge] = ROCK

        return Grid(new_grid)

    def print_autocorrelation(self):
        if not self.row_corr or not self.col_corr:
            self.autocorrelation()

        lines = []
        lines.append("row autocorrelation")
        for r1 in range(self.nrow):
            line = []
            for r2 in range(self.nrow):
                line.append(f"{self.row_corr[r1][r2]:2d}")
            lines.append(" ".join(line))
        lines.append("")

        lines.append("column autocorrelation")
        for c1 in range(self.ncol):
            line = []
            for c2 in range(self.ncol):
                line.append(f"{self.col_corr[c1][c2]:2d}")
            lines.append(" ".join(line))
        lines.append("")
        print("\n".join(lines))

    def autocorrelation(self):
        self.row_corr = []
        for r1 in range(self.nrow):
            row = []
            for r2 in range(self.nrow):
                a12 = sum([int(a == b) for a, b in zip(self.row[r1], self.row[r2])])
                row.append(a12)
            self.row_corr.append(row)

        self.col_corr = []
        for c1 in range(self.ncol):
            row = []
            for c2 in range(self.ncol):
                a12 = sum([int(a == b) for a, b in zip(self.col[c1], self.col[c2])])
                row.append(a12)
            self.col_corr.append(row)

    def possible_smudges(self):
        if not self.row_corr or not self.col_corr:
            self.autocorrelation()

        result = []

        smudge = self.ncol - 1
        for r1 in range(self.nrow-1):
            for r2 in range(r1+1, self.nrow):
                if self.row_corr[r1][r2] == smudge:
                    for col in range(self.ncol):
                        if self.row[r1][col] != self.row[r2][col]:
                            result.append((r1, col))
                            result.append((r2, col))

        smudge = self.nrow - 1
        for c1 in range(self.ncol-1):
            for c2 in range(c1+1, self.ncol):
                if self.col_corr[c1][c2] == smudge:
                    for row in range(self.nrow):
                        if self.col[c1][row] != self.col[c2][row]:
                            result.append((row, c1))
                            result.append((row, c2))

        return result


    def reflection(self, smudges=False) -> Tuple[int, int]:
        r1 = self.row_reflection1()
        r2 = self.row_reflection2()
        c1 = self.col_reflection1()
        c2 = self.col_reflection2()
        return r1, r2, c1, c2

    def row_reflection1(self) -> int:
        locs = findall(self.row, self.row[0])
        if len(locs) < 2:
            return 0
        for end in locs[-1:0:-1]:
            if (end - 1) % 2 == 0:
                if self.is_row_reflection(0, end):
                    return (end + 1) // 2
        return 0

    def row_reflection2(self) -> int:
        locs = findall(self.row, self.row[-1])
        if len(locs) < 2:
            return 0

        for start in locs[:-1]:
            if (self.nrow - start) % 2 == 0:
                if self.is_row_reflection(start, self.nrow-1):
                    return start + (self.nrow - start) // 2
        return 0

    def is_row_reflection(self, start, end) -> bool:
        for i in range(end-start+1//2):
            if self.row[start+i] != self.row[end-i]:
                return False
        return True

    def col_reflection1(self) -> int:
        locs = findall(self.col, self.col[0])
        if len(locs) < 2:
            return 0
        for end in locs[-1:0:-1]:
            if (end - 1) % 2 == 0:
                if self.is_col_reflection(0, end):
                    return (end + 1) // 2
        return 0

    def col_reflection2(self) -> int:
        locs = findall(self.col, self.col[-1])
        if len(locs) < 2:
            return 0

        for start in locs[:-1]:
            if (self.ncol - start) % 2 == 0:
                if self.is_col_reflection(start, self.ncol-1):
                    return start + (self.ncol - start) // 2
        return 0

    def is_col_reflection(self, start, end) -> bool:
        for i in range(end-start+1//2):
            if self.col[start+i] != self.col[end-i]:
                return False
        return True

def findall(items, target) -> list[int]:
    """Return list of indices where target is found in items."""
    result = []
    if items[0] == target:
        result.append(0)
    start = 0
    while start > -1:
        try:
            start = items.index(target, start+1)
            if start > -1:
                result.append(start)
        except ValueError:
            start = -1
    return result


def parse_input(lines):
    maps = []
    for sect in parse_sections(lines):
        maps.append(Grid([list(line.strip()) for line in sect]))
    return maps

def solve2(lines: Lines) -> int:
    """Solve the problem."""
    patterns = parse_input(lines)
    result = 0
    prev_result = 0
    for num, patt in enumerate(patterns):
        # print(f"\npattern {num}...")
        # print(str(patt))
        # print()
        # patt.print_autocorrelation()
        rmirror1, rmirror2, cmirror1, cmirror2 = patt.reflection()
        score = max(cmirror1, cmirror2) + 100*max(rmirror1, rmirror2)
        prev_result += score
        # print(f"rmirror, cmirror: {rmirror1}, {rmirror2}, {cmirror1}, {cmirror2}")
        score = 0
        for rsmudge, csmudge in patt.possible_smudges():
            # print(f"rsmudge, csmudge: {rsmudge}, {csmudge}...")
            fixed_patt = patt.fix_smudge(rsmudge, csmudge)
            # print(str(fixed_patt))
            rfix1, rfix2, cfix1, cfix2 = fixed_patt.reflection()
            # print(f"rsmudge, csmudge: {rsmudge}, {csmudge} --> rfix, cfix: {rfix1}, {rfix2}, {cfix1}, {cfix2}")
            if rfix1 or rfix2 or cfix1 or cfix2:
                if rfix1 and rfix1 != rmirror1:
                    score = 100 * rfix1
                    break
                elif rfix2 and rfix2 != rmirror2:
                    score = 100 * rfix2
                    break
                elif cfix1 and cfix1 != cmirror1:
                    score = cfix1
                    break
                elif cfix2 and cfix2 != cmirror2:
                    score = cfix2
                    break
        if score == 0:
            raise RuntimeError("No smudge was found to fix!")
        result += score
    return result

def solve(lines: Lines) -> int:
    """Solve the problem."""
    patterns = parse_input(lines)
    result = 0
    for patt in patterns:
        r1, r2, c1, c2 = patt.reflection()
        score = max(c1, c2) + 100*max(r1, r2)
        result += score
    return result


# PART 1

def example1() -> None:
    """Run example for problem with input arguments."""
    print("EXAMPLE 1:")
    for text, expected in SAMPLE_CASES:
        lines = load_text(text, blank_lines=True)
        result = solve(lines)
        print(f"'{text}' -> {result} (expected {expected})")
        assert result == expected
    print("= " * 32)

def part1(lines: Lines) -> None:
    print("PART 1:")
    result = solve(lines)
    print(f"result is {result}")
    assert result == 30158
    print("= " * 32)


# PART 2

def example2() -> None:
    """Run example for problem with input arguments."""
    print("EXAMPLE 2:")
    for text, expected in SAMPLE_CASES2:
        lines = load_text(text, blank_lines=True)
        result = solve2(lines)
        print(f"'{text}' -> {result} (expected {expected})")
        assert result == expected
    print("= " * 32)

def part2(lines: Lines) -> None:
    print("PART 2:")
    result = solve2(lines)
    print(f"result is {result}")
    assert result == 36474
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE, blank_lines=True)
    part1(input_lines)
    example2()
    part2(input_lines)
