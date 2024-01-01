#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 21
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
...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
...........
        """, 6, 16
    ),
]

SAMPLE_CASES2 = [
    (
        """
...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
...........
        """, 500, 167004
    ),
]

SAMPLE_CASES3 = [
    (
        """
...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
...........
        """, 6, 16
    ),
    (
        """
...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
...........
        """, 10, 50
    ),
    (
        """
...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
...........
        """, 50, 1594
    ),
    (
        """
...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
...........
        """, 100, 6536
    ),
    (
        """
...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
...........
        """, 500, 167004
    ),
    (
        """
...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
...........
        """, 1000, 668697
    ),
    (
        """
...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
...........
        """, 5000, 16733044
    ),
]


Lines = Sequence[str]
Sections = Sequence[Lines]

# Utility functions

def load_input(infile: str, strip=True, blank_lines=False) -> Lines:
    return load_text(
        Path(infile).read_text(), strip=strip, blank_lines=blank_lines
    )

def load_text(text: str, strip=True, blank_lines=False) -> Lines:
    if strip:
        lines = [line.strip() for line in text.strip("\n").split("\n")]
    else:
        lines = text.strip("\n").split("\n")
    if blank_lines:
        return lines
    return [line for line in lines if line.strip()]


# Solution

Direction = str

UP, DOWN, RIGHT, LEFT = "U", "D", "R", "L"
DIRECTIONS = (UP, DOWN, RIGHT, LEFT)
OPPOSITE = {UP: DOWN, RIGHT: LEFT, DOWN: UP, LEFT: RIGHT}

GARDEN, ROCK, START = ".", "#", "S"


Grid = dict["Pos", str]

@dataclass(order=True, frozen=True)
class Pos():
    row: int
    col: int

    def __str__(self) -> str:
        return f"({self.row},{self.col})"

    def dist(self, other: "Pos") -> int:
        return abs(other.row - self.row) + abs(other.col - self.col)

    def neighbors(self) -> "List[Pos]":
        return [
            Pos(self.row, self.col + 1),
            Pos(self.row + 1, self.col),
            Pos(self.row, self.col - 1),
            Pos(self.row - 1, self.col),
        ]

    def neighbor(self, direction: Direction, dist: int = 1) -> "Pos":
        if direction == UP:
            return Pos(self.row - dist, self.col)
        if direction == RIGHT:
            return Pos(self.row, self.col + dist)
        if direction == DOWN:
            return Pos(self.row + dist, self.col)
        if direction == LEFT:
            return Pos(self.row, self.col - dist)
        raise ValueError(f"Unrecognized direction '{direction}'")


class Board:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.rowmax = max((v.row for v in self.grid.keys()))
        self.rowmin = min((v.row for v in self.grid.keys()))
        self.colmax = max((v.col for v in self.grid.keys()))
        self.colmin = min((v.col for v in self.grid.keys()))

        self.start = [pos for pos, item in self.grid.items() if item == START][0]
        self.grid[self.start] = GARDEN
        assert self.start

    @classmethod
    def from_lines(cls, lines: list[str]) -> "Board":
        grid = defaultdict(lambda: ROCK)
        for row, line in enumerate(lines):
            for col, item in enumerate(line):
                grid[Pos(row, col)] = item
        return cls(grid)

    def __str__(self) -> str:
        lines = []
        for row in range(self.rowmin, self.rowmax+1):
            line = []
            for col in range(self.colmin, self.colmax+1):
                line.append(str(self.at(Pos(row, col))))
            lines.append("".join(line))
        return "\n".join(lines)

    def at(self, pos):
        return self.grid[pos]

    def explore(self, max_steps=2) -> int:
        q = deque([])
        q.append((0, self.start))

        locs = set()
        visited = set()
        visited.add((0, self.start))
        while q:
            steps, pos = q.popleft()
            # print(f">>> {steps}:  {pos}")
            if steps > max_steps:
                break
            if steps == max_steps:
                locs.add(pos)
            for naypos in pos.neighbors():
                if self.at(naypos) == GARDEN and (steps+1, naypos) not in visited:
                    visited.add((steps+1, naypos))
                    q.append((steps+1, naypos))

        return len(locs)


class Board2:
    def __init__(self, grid: Grid):
        self.grid = grid
        assert min((v.row for v in self.grid.keys())) == 0
        assert min((v.col for v in self.grid.keys())) == 0

        self.nrow = max((v.row for v in self.grid.keys())) + 1
        self.ncol = max((v.col for v in self.grid.keys())) + 1

        self.start = [pos for pos, item in self.grid.items() if item == START][0]
        assert self.start

        self.grid[self.start] = GARDEN

    @classmethod
    def from_lines(cls, lines: list[str]) -> "Board2":
        grid = defaultdict(lambda: ROCK)
        for row, line in enumerate(lines):
            for col, item in enumerate(line):
                grid[Pos(row, col)] = item
        return cls(grid)

    def __str__(self) -> str:
        lines = []
        for row in range(self.nrow):
            line = []
            for col in range(self.ncol):
                line.append(str(self.at(Pos(row, col))))
            lines.append("".join(line))
        return "\n".join(lines)

    def at(self, pos):
        row = pos.row % self.nrow
        col = pos.col % self.ncol
        return self.grid[Pos(row, col)]

    def explore(self, max_steps=2) -> int:
        q = deque([])
        q.append((0, self.start))

        locs = set()
        visited = set()
        visited.add((0, self.start))
        nplots = [0] * (max_steps + 1) # number of plots reached after 'steps' steps
        last_steps = -1
        while q:
            steps, pos = q.popleft()
            # print(f">>> {steps}:  {pos}")
            if steps > max_steps:
                break
            if steps == max_steps:
                locs.add(pos)
            nplots[steps] += 1
            if steps and steps > last_steps:
                print(f"{last_steps},{nplots[last_steps]}")
            for naypos in pos.neighbors():
                if self.at(naypos) == GARDEN and (steps+1, naypos) not in visited:
                    visited.add((steps+1, naypos))
                    q.append((steps+1, naypos))
            last_steps = steps

        return len(locs)



def solve2(lines: Lines, max_steps=64) -> int:
    """Solve the problem."""
    board = Board2.from_lines(lines)
    print(board)
    print()

    # visited = board.explore(max_steps)
    visited = board.explore(750)
    return visited

def solve(lines: Lines, max_steps=64) -> int:
    """Solve the problem."""
    board = Board.from_lines(lines)
    print(board)
    print()

    visited = board.explore(max_steps)
    return visited


# PART 1

def example1() -> None:
    """Run example for problem with input arguments."""
    print("EXAMPLE 1:")
    for text, steps, expected in SAMPLE_CASES:
        lines = load_text(text)
        result = solve(lines, steps)
        print(f"'{text}' -> {result} (expected {expected})")
        assert result == expected
    print("= " * 32)

def part1(lines: Lines) -> None:
    print("PART 1:")
    result = solve(lines, 64)
    print(f"result is {result}")
    assert result == 3594
    print("= " * 32)


# PART 2

def example2() -> None:
    """Run example for problem with input arguments."""
    print("EXAMPLE 2:")
    for text, steps, expected in SAMPLE_CASES2:
        lines = load_text(text)
        result = solve2(lines, steps)
        print(f"'{text}', {steps} -> {result} (expected {expected})")
        assert result == expected
    print("= " * 32)

def part2(lines: Lines) -> None:
    print("PART 2:")
    result = solve2(lines, 26501365)
    print(f"result is {result}")
    assert result == -1
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    example2()
    part2(input_lines)
