#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 10
#
from typing import Sequence, Union, Optional, Any, Dict, List, Tuple
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass
from pprint import pprint
import math
import re

INPUTFILE = "input.txt"

SAMPLE_CASES = [
    (
        """
        .....
        .S-7.
        .|.|.
        .L-J.
        .....
        """,
        4
    ),
    (
        """
        ..F7.
        .FJ|.
        SJ.L7
        |F--J
        LJ...
        """,
        8
    ),
]

SAMPLE_CASES2 = [
    (
        """
        .....
        .S-7.
        .|.|.
        .L-J.
        .....
        """,
        1
    ),
    (
        """
        ..F7.
        .FJ|.
        SJ.L7
        |F--J
        LJ...
        """,
        1
    ),
    (
        """
        ...........
        .S-------7.
        .|F-----7|.
        .||.....||.
        .||.....||.
        .|L-7.F-J|.
        .|..|.|..|.
        .L--J.L--J.
        ...........
        """,
        4
    ),
    (
        """
        ..........
        .S------7.
        .|F----7|.
        .||....||.
        .||....||.
        .|L-7F-J|.
        .|..||..|.
        .L--JL--J.
        ..........
        """,
        4
    ),
    (
        """
        .F----7F7F7F7F-7....
        .|F--7||||||||FJ....
        .||.FJ||||||||L7....
        FJL7L7LJLJ||LJ.L-7..
        L--J.L7...LJS7F-7L7.
        ....F-J..F7FJ|L7L7L7
        ....L7.F7||L7|.L7L7|
        .....|FJLJ|FJ|F7|.LJ
        ....FJL-7.||.||||...
        ....L---J.LJ.LJLJ...
        """,
        8
    ),
    (
        """
        FF7FSF7F7F7F7F7F---7
        L|LJ||||||||||||F--J
        FL-7LJLJ||||||LJL-77
        F--JF--7||LJLJ7F7FJ-
        L---JF-JLJ.||-FJLJJ7
        |F|F-JF---7F7-L7L|7|
        |FFJF7L7F-JF7|JL---7
        7-L-JL7||F7|L7F-7F7|
        L.L7LFJ|||||FJL7||LJ
        L7JLJL-JLJLJL--JLJ.L
        """,
        10
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


GROUND, START = ".", "S"
NS, EW, NE, NW, SW, SE = "|", "-", "L", "J", "7", "F"

@dataclass
class Grid:
    grid: dict[Pos, str]
    start: Optional[Pos] = None

    def __post_init__(self):
        for pos, ch in self.grid.items():
            # print(f">> @{pos} is {ch}")
            if ch == START:
                self.start = pos
            else:
                assert ch in (NS, EW, NE, NW, SE, SW)

        if self.start:
            pos = self.start
            north, south, east, west = False, False, False, False
            if self.at(pos.north()) in (NS, SW, SE):
                north = True
            if self.at(pos.south()) in (NS, NW, NE):
                south = True
            if self.at(pos.east()) in (EW, NW, SW):
                east = True
            if self.at(pos.west()) in (EW, NE, SE):
                west = True
            if north and south:
                self.grid[pos] = NS
            elif east and west:
                self.grid[pos] = EW
            elif north and east:
                self.grid[pos] = NE
            elif north and west:
                self.grid[pos] = NW
            elif south and east:
                self.grid[pos] = SE
            elif south and west:
                self.grid[pos] = SW
            else:
                raise ValueError("Cannot assign unique pipe segment to {pos}")

    def __str__(self):
        lines = []
        rmin, rmax, cmin, cmax = self.bounds()
        for r in range(rmin-1, rmax+2):
            line = []
            for c in range(cmin-1, cmax+2):
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
        return self.grid.get(pos, GROUND)

    def neighbors(self, pos) -> list[Pos]:
        if self.at(pos) == NS:
            return (pos.north(), pos.south())
        elif self.at(pos) == EW:
            return (pos.east(), pos.west())
        elif self.at(pos) == NE:
            return (pos.north(), pos.east())
        elif self.at(pos) == NW:
            return (pos.north(), pos.west())
        elif self.at(pos) == SE:
            return (pos.south(), pos.east())
        elif self.at(pos) == SW:
            return (pos.south(), pos.west())
        raise RuntimeError(f"@{pos} is not in pipeline")

    def find_loop(self) -> list[Pos]:
        # print(f"Start @{self.start}")

        pq = deque([(1, pos) for pos in self.neighbors(self.start)])
        visited = set([self.start])
        while len(pq):
            dist, pos = pq.popleft()
            # print(f"pop @{pos} dist {dist}")
            for naypos in self.neighbors(pos):
                if naypos not in visited:
                    pq.append((dist+1, naypos))
                    visited.add(naypos)
                    # print(f"-- @{naypos} dist {dist+1}")
        return list(visited)


def parse_input(lines):
    grid = defaultdict(str)
    for row, line in enumerate(lines):
        assert line
        for col, ch in enumerate(line):
            if ch != GROUND:
                grid[Pos(row, col)] = ch
    return Grid(grid)


def solve2(lines: Lines) -> int:
    """Solve the problem."""
    grid = parse_input(lines)
    print(grid)
    loop = grid.find_loop()
    outside = set()
    inside = set()

    rmin, rmax, cmin, cmax = grid.bounds()

    count = 0
    for row in range(rmin-1, rmax+2):
        out = True
        north, south = False, False
        for col in range(cmin-1, cmax+2):
            pos = Pos(row, col)
            if pos in loop:
                if grid.at(pos) == NS:
                    north, south = not north, not south
                elif grid.at(pos) in (NE, NW):
                    north = not north
                elif grid.at(pos) in (SE, SW):
                    south = not south
                if north and south:
                    north, south = False, False
                    out = not out
                # print(f"loop@{pos} {grid.at(pos)} outside:{out} | north:{north} south:{south}")
            else:
                if not out:
                    count += 1
                # print(f"--- @{pos} {grid.at(pos)} outside:{out} | north:{north} south:{south}")

    return count


def solve(lines: Lines) -> int:
    """Solve the problem."""
    grid = parse_input(lines)
    print(grid)
    loop = grid.find_loop()
    return math.ceil(len(loop) / 2)


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
    assert result == 6890
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
    assert result == 453
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    example2()
    part2(input_lines)
