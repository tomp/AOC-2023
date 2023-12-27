#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 16
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
        r"""
        .|...\....
        |.-.\.....
        .....|-...
        ........|.
        ..........
        .........\
        ..../.\\..
        .-.-/..|..
        .|....-|.\
        ..//.|....
        """,
        46
    ),
]


SAMPLE_CASES2 = [
    (
        r"""
        .|...\....
        |.-.\.....
        .....|-...
        ........|.
        ..........
        .........\
        ..../.\\..
        .-.-/..|..
        .|....-|.\
        ..//.|....
        """,
        51
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
        lines = [line for line in text.strip("\n").split("\n")]
    if blank_lines:
        return lines
    return [line for line in lines if line.strip()]

# Solution

NORTH, SOUTH, EAST, WEST = "N", "S", "E", "W"

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

    def neighbor(self, direction) -> "Pos":
        if direction == NORTH:
            return self.north()
        if direction == EAST:
            return self.east()
        if direction == SOUTH:
            return self.south()
        if direction == WEST:
            return self.west()
        raise ValueError(f"Unrecognized direction '{direction}'")


EMPTY, VSPLIT, HSPLIT, RMIRROR, LMIRROR = ".", "|", "-", "/", "\\"

class Grid:
    def __init__(self, grid):
        self.grid = grid

        assert all([ch in (VSPLIT, HSPLIT, RMIRROR, LMIRROR) for ch in grid.values()])
        self.nrow = 1 + max([pos.row for pos in self.grid.keys()])
        self.ncol = 1 + max([pos.col for pos in self.grid.keys()])

    def __str__(self):
        lines = []
        for r in range(self.nrow):
            line = []
            for c in range(self.ncol):
                line.append(self.at(Pos(r, c)))
            lines.append("".join(line))
        return "\n".join(lines)

    def at(self, pos: Pos) -> str:
        return self.grid[pos]

    def on_grid(self, pos) -> bool:
        return (0 <= pos.row < self.nrow) and (0 <= pos.col < self.ncol)

    def propagate(self, pos: Pos, direction: str) -> set[Pos]:
        active = set()
        visited = set()

        active.add(pos)
        beams = []
        for next_direction in new_direction(direction, self.at(pos)):
            beams.append((pos, next_direction))

        steps = 0
        while beams:
            # print(f"step {steps:03d}: {', '.join([f'[{p} {d}]' for p, d in beams])}")
            next_beams = []
            for pos, direction in beams:
                visited.add((pos, direction))

                next_pos = pos.neighbor(direction)
                if not self.on_grid(next_pos):
                    continue
                active.add(next_pos)

                tile = self.at(next_pos)
                for next_direction in new_direction(direction, tile):
                    if (next_pos, next_direction) not in visited:
                        next_beams.append((next_pos, next_direction))
            beams = next_beams
            steps += 1

        return active


def new_direction(direction, tile) ->  list[str]:
    if tile == EMPTY:
        return (direction,)
    elif tile == RMIRROR:
        if direction == NORTH:
            return (EAST,)
        elif direction == SOUTH:
            return (WEST,)
        elif direction == EAST:
            return (NORTH,)
        elif direction == WEST:
            return (SOUTH,)
    elif tile == LMIRROR:
        if direction == NORTH:
            return (WEST,)
        elif direction == SOUTH:
            return (EAST,)
        elif direction == EAST:
            return (SOUTH,)
        elif direction == WEST:
            return (NORTH,)
    elif tile == HSPLIT:
        if direction == NORTH or direction == SOUTH:
            return (WEST, EAST)
        else:
            return (direction)
    elif tile == VSPLIT:
        if direction == EAST or direction == WEST:
            return (NORTH, SOUTH)
        else:
            return (direction)
    else:
        raise ValueError(f"Unsupported tile '{tile}'")


def parse_input(lines):
    grid = defaultdict(lambda: EMPTY)
    for row, line in enumerate(lines):
        for col, ch in enumerate(line):
            if ch != EMPTY:
                grid[Pos(row, col)] = ch
    return Grid(grid)

def solve2(lines: Lines) -> int:
    """Solve the problem."""
    grid = parse_input(lines)
    print(f"{grid.nrow} row, {grid.ncol} columns")

    result = 0

    for row in range(grid.nrow):
        active = grid.propagate(Pos(row, 0), EAST)
        if len(active) > result:
            result = len(active)

        active = grid.propagate(Pos(row, grid.ncol - 1), WEST)
        if len(active) > result:
            result = len(active)

    for col in range(grid.ncol):
        active = grid.propagate(Pos(0, col), SOUTH)
        if len(active) > result:
            result = len(active)

        active = grid.propagate(Pos(grid.nrow - 1, col), NORTH)
        if len(active) > result:
            result = len(active)

    return result

def solve(lines: Lines) -> int:
    """Solve the problem."""
    grid = parse_input(lines)
    print(grid)
    print(f"{grid.nrow} row, {grid.ncol} columns")

    active = grid.propagate(Pos(0,0), EAST)

    return len(active)


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
    assert result == 8125
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
    assert result == 8489
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    example2()
    part2(input_lines)
