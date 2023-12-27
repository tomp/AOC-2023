#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 17
#
from typing import Sequence, Union, Optional, Any, Dict, List, Tuple
from pathlib import Path
from collections import defaultdict, deque
from heapq import heapify, heappush, heappop
from dataclasses import dataclass
from pprint import pprint
import math
import re

INPUTFILE = "input.txt"

SAMPLE_CASES = [
    (
        """
        24134323113
        32154535356
        32552456542
        """,
        43
    ),
    (
        """
        2413432311323
        3215453535623
        3255245654254
        3446585845452
        4546657867536
        1438598798454
        4457876987766
        3637877979653
        """,
        73
    ),
    (
        """
        2413432311323
        3215453535623
        3255245654254
        3446585845452
        4546657867536
        1438598798454
        4457876987766
        3637877979653
        4654967986887
        4564679986453
        1224686865563
        2546548887735
        4322674655533
        """,
        102
    ),
]

SAMPLE_CASES2 = [
    (
        """
        2413432311323
        3215453535623
        3255245654254
        3446585845452
        4546657867536
        1438598798454
        4457876987766
        3637877979653
        4654967986887
        4564679986453
        1224686865563
        2546548887735
        4322674655533
        """,
        94
    ),
    (
        """
        111111111111
        999999999991
        999999999991
        999999999991
        999999999991
        """,
        71
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
DIRECTIONS = (NORTH, SOUTH, EAST, WEST)

REVERSE = { NORTH: SOUTH, EAST: WEST, SOUTH: NORTH, WEST: EAST }
ARROW = { NORTH: "^", EAST: ">", SOUTH: "v", WEST: "<" }


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


Grid = dict[Pos, int]
Direction = str
State = Tuple[Pos, Direction, int, int]  # pos, direction, run, heatloss
Step = Tuple[Pos, Direction]
History = list[Step]

class Board:
    def __init__(self, blocks: Grid):
        self.grid = blocks
        self.rowmax = max([v.row for v in self.grid.keys()])
        self.rowmin = min([v.row for v in self.grid.keys()])
        self.colmax = max([v.col for v in self.grid.keys()])
        self.colmin = min([v.col for v in self.grid.keys()])

    @classmethod
    def from_lines(cls, lines: list[str]) -> "Board":
        grid = defaultdict(lambda: 1000)
        for row, line in enumerate(lines):
            for col, ch in enumerate(line):
                grid[Pos(row, col)] = int(ch)
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

    def heatloss(self, hist: History) -> int:
        return sum([self.at(pos) for pos, _ in hist])

    def print(self, hist: Optional[History] = None):
        path = defaultdict(str)
        if hist:
            heatloss = 0
            for pos, direction in hist:
                path[pos] = direction
                heatloss += self.at(pos)
            steps = len(hist)

        lines = []
        for row in range(self.rowmin, self.rowmax+1):
            line = []
            for col in range(self.colmin, self.colmax+1):
                pos = Pos(row, col)
                if pos in path:
                    line.append(ARROW[path[pos]])
                else:
                    line.append(str(self.at(Pos(row, col))))
            lines.append("".join(line))
        print("\n".join(lines))
        if hist:
            print(f"Steps: {len(hist)}  Total heat loss: {heatloss}")
        print()


def astar_search_with_history(board: Board, maxrun=10, minrun=4) -> History:

    start = Pos(board.rowmin, board.colmin)
    target = Pos(board.rowmax, board.colmax)
    dist = start.dist(target)

    q = []
    visited = defaultdict(lambda: 1000000000)

    heappush(q, (dist, start, EAST, 0, 0, []))
    visited[(start, EAST, 0, 0)] = 0

    while q:
        cost, pos, direction, run, heatloss, hist = heappop(q)
        # print(f">>> {cost}, {heatloss} : @{pos} {direction}*{run}  {', '.join([f'@{pos} {direction}' for pos, direction in reversed(hist[:-1])])}")
        if pos == target and run >= minrun:
            return hist

        for next_direction in DIRECTIONS:
            if next_direction == direction:
                if run >= maxrun:
                    continue
                next_run = run + 1
            elif run < minrun:
                continue
            elif next_direction == REVERSE[direction]:
                continue
            else:
                next_run = 1

            next_pos = pos.neighbor(next_direction)
            if board.at(next_pos) > 9:
                continue
            next_heatloss = heatloss + board.at(next_pos)
            next_dist = next_pos.dist(target)
            if visited[(next_pos, next_direction, next_run)] > next_heatloss:
               heappush(q,
                   (next_dist + next_heatloss, next_pos, next_direction, next_run, next_heatloss,
                            hist + [(next_pos, next_direction)]))
               visited[(next_pos, next_direction, next_run)] = next_heatloss

    return []

def astar_search(board: Board, maxrun=3, minrun=0) -> int:

    start = Pos(board.rowmin, board.colmin)
    target = Pos(board.rowmax, board.colmax)
    dist = start.dist(target)

    q = []
    visited = defaultdict(lambda: 1000000000)

    heappush(q, (dist, start, EAST, 0, 0))
    visited[(start, EAST, 0, 0)] = 0

    while q:
        cost, pos, direction, run, heatloss = heappop(q)
        # print(f">>> {cost}, {heatloss} : @{pos} {direction}*{run}  {', '.join([f'@{pos} {direction}' for pos, direction in reversed(hist[:-1])])}")
        if pos == target and run >= minrun:
            return heatloss

        for next_direction in DIRECTIONS:
            if next_direction == direction:
                if run >= maxrun:
                    continue
                next_run = run + 1
            elif run < minrun:
                continue
            elif next_direction == REVERSE[direction]:
                continue
            else:
                next_run = 1

            next_pos = pos.neighbor(next_direction)
            if board.at(next_pos) > 9:
                continue
            next_heatloss = heatloss + board.at(next_pos)
            next_dist = next_pos.dist(target)
            if visited[(next_pos, next_direction, next_run)] > next_heatloss:
               heappush(q,
                   (next_dist + next_heatloss, next_pos, next_direction, next_run, next_heatloss))
               visited[(next_pos, next_direction, next_run)] = next_heatloss

    return []



def solve2(lines: Lines) -> int:
    """Solve the problem."""
    board = Board.from_lines(lines)
    board.print()

    # heatloss = astar_search2(board, 10, 4)
    # return heatloss

    hist = astar_search_with_history(board, 10, 4)
    board.print(hist)
    return board.heatloss(hist)

def solve(lines: Lines) -> int:
    """Solve the problem."""
    board = Board.from_lines(lines)
    board.print()

    # heatloss = astar_search(board, 3, 0)
    # return heatloss

    hist = astar_search_with_history(board, 3, 0)
    board.print(hist)
    return board.heatloss(hist)


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
    assert result == 1099
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
    assert result == 1266
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    example2()
    part2(input_lines)
