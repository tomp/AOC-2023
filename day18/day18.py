#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 18
#
from typing import Sequence, Iterable, Union, Optional, Any, Dict, List, Tuple
from pathlib import Path
from collections import defaultdict, deque
from itertools import combinations, permutations, groupby
from dataclasses import dataclass
from pprint import pprint
import math
import re

INPUTFILE = "input.txt"

SAMPLE_CASES = [
    (
        """
        R 6 (#70c710)
        D 5 (#0dc571)
        L 2 (#5713f0)
        D 2 (#d2c081)
        R 2 (#59c680)
        D 2 (#411b91)
        L 5 (#8ceee2)
        U 2 (#caa173)
        L 1 (#1b58a2)
        U 2 (#caa171)
        R 2 (#7807d2)
        U 3 (#a77fa3)
        L 2 (#015232)
        U 2 (#7a21e3)
        """,
        62
    ),
]

SAMPLE_CASES2 = [
    (
        """
        R 6 (#70c710)
        D 5 (#0dc571)
        L 2 (#5713f0)
        D 2 (#d2c081)
        R 2 (#59c680)
        D 2 (#411b91)
        L 5 (#8ceee2)
        U 2 (#caa173)
        L 1 (#1b58a2)
        U 2 (#caa171)
        R 2 (#7807d2)
        U 3 (#a77fa3)
        L 2 (#015232)
        U 2 (#7a21e3)
        """,
        952408144115
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

Grid = dict["Pos", int]
Grid2 = dict["Pos", list["Corner"]]
Direction = str

UP, DOWN, RIGHT, LEFT = "U", "D", "R", "L"
DIRECTIONS = (UP, DOWN, RIGHT, LEFT)
OPPOSITE = {UP: DOWN, RIGHT: LEFT, DOWN: UP, LEFT: RIGHT}
DIR_TO_CODE = {RIGHT: 0, DOWN: 1, LEFT: 2, UP: 3}
CODE_TO_DIR = {code: direction for direction, code in DIR_TO_CODE.items()}

PLAN_RE = re.compile(r"([RLUD]) (\d+) \(#([a-f0-9]{6})\)")
EMPTY, TRENCH = ".", "#"


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

    def all_neighbors(self) -> "List[Pos]":
        return [
            Pos(self.row - 1, self.col - 1),
            Pos(self.row - 1, self.col),
            Pos(self.row - 1, self.col + 1),
            Pos(self.row,     self.col + 1),
            Pos(self.row + 1, self.col + 1),
            Pos(self.row + 1, self.col),
            Pos(self.row + 1, self.col - 1),
            Pos(self.row,     self.col - 1),
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
    def __init__(self, blocks: Grid):
        self.grid = blocks
        self.rowmax = max((v.row for v in self.grid.keys()))
        self.rowmin = min((v.row for v in self.grid.keys()))
        self.colmax = max((v.col for v in self.grid.keys()))
        self.colmin = min((v.col for v in self.grid.keys()))

    @classmethod
    def from_plan(cls, plan: Sequence[Tuple[str, int]]) -> "Board":
        grid = defaultdict(lambda: EMPTY)
        pos = Pos(0, 0)
        grid[pos] = TRENCH
        for direction, length in plan:
            for _ in range(int(length)):
                pos = pos.neighbor(direction)
                grid[pos] = TRENCH
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

    def onboard(self, pos: Pos) -> bool:
        return (
            (self.rowmin <= pos.row <= self.rowmax) and
            (self.colmin <= pos.col <= self.colmax)
        )

    def volume(self) -> int:
        return len([pos for pos, item in self.grid.items() if item == TRENCH])

    def fill(self, start: Pos) -> bool:
        if self.at(start) == TRENCH:
            return False

        q = deque([])
        q.append(start)

        visited = set()
        while q:
            pos = q.popleft()
            for naypos in pos.neighbors():
                if not self.onboard(naypos):
                    return False
                if self.at(naypos) == EMPTY and naypos not in visited:
                    visited.add(naypos)
                    q.append(naypos)

        for pos in visited:
            self.grid[pos] = TRENCH
        return True


class Corner:
    def __init__(self, pos: Pos = None, rl: str = "", ud: str = "", length: int = 0):
        self.pos = pos
        self.rl = rl  # right-left
        self.ud = ud  # up-down
        self.length = length

    def __str__(self) -> str:
        return f"{self.pos}-{self.ud}{self.rl}-{self.length}"


@dataclass
class Segment:
    start: int
    end: int

    def __str__(self) -> str:
        return f"[{self.start} - {self.end}]"

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class Board2:
    def __init__(self, corners: Grid2):
        self.grid = corners
        self.rowmax = max((v.row for v in self.grid.keys()))
        self.rowmin = min((v.row for v in self.grid.keys()))
        self.colmax = max((v.col for v in self.grid.keys()))
        self.colmin = min((v.col for v in self.grid.keys()))

    def __str__(self) -> str:
        lines = []
        ncol = self.colmax - self.colmin + 1
        for row, locs in self.rows():
            line = []
            for pos in locs:
                line.append(str(self.grid[pos]))
            lines.append(", ".join(line))
        return "\n".join(lines)

    @classmethod
    def from_plan(cls, plan: list[Tuple[str, int]]) -> "Board2":
        grid = defaultdict(Corner)
        pos = Pos(0, 0)
        for direction, length in plan:
            # print(f"{direction} {length}")

            corner = grid[pos]
            corner.pos = pos
            if direction in (RIGHT, LEFT):
                corner.rl = direction
                corner.length = length
            elif direction in (UP, DOWN):
                corner.ud = direction

            pos = pos.neighbor(direction, length)
            corner2 = grid[pos]
            corner2.pos = pos
            if direction in (RIGHT, LEFT):
                corner2.rl = OPPOSITE[direction]
                corner2.length = length
            elif direction in (UP, DOWN):
                corner2.ud = OPPOSITE[direction]

        return cls(grid)

    def rows(self) -> Iterable[Tuple[int, Iterable[Pos]]]:
        return groupby(
            sorted(self.grid.keys(), key=lambda p: (p.row, p.col)),
            key=lambda p: p.row
        )

    def volume(self) -> int:
        result = 0
        segments = []
        last_row = self.rowmin

        for row, locs in self.rows():
            # print(f"\nrow {row:6d}:")
            # print(f"  segments:  {', '.join(map(str, segments))}")

            corners = [self.grid[pos] for pos in locs]
            # print(f"   corners:  {', '.join(map(str, corners))}")

            result += (row - last_row) * sum([seg.length for seg in segments])
            # print(f"          :  {result} <- += {row - last_row} * {sum([seg.length for seg in segments])}")

            last_row = row

            if not segments:
                for corner, corner2 in pairs(corners):
                    assert corner.pos.col < corner2.pos.col
                    assert corner.rl == RIGHT and corner2.rl == LEFT
                    segments.append(Segment(corner.pos.col, corner2.pos.col))
                last_row = row
                continue

            iseg = 0
            seg = segments[iseg]
            for corner, corner2 in pairs(corners):
                assert corner.pos.col < corner2.pos.col
                assert corner.rl == RIGHT and corner2.rl == LEFT

                while corner.pos.col > seg.end and iseg + 1 < len(segments):
                    iseg += 1
                    if iseg < len(segments):
                        seg = segments[iseg]
                # print(f"      >>> iseg: {iseg}  seg: {seg}  corner: {corner}  corner2: {corner2}")

                if corner2.pos.col < seg.start:
                    assert corner.ud == DOWN and corner2.ud == DOWN
                    segments = segments[:iseg] + [Segment(corner.pos.col, corner2.pos.col)] + segments[iseg:]
                    iseg += 1
                    seg = segments[iseg]

                elif corner2.pos.col == seg.start:
                    assert corner.ud == DOWN and corner2.ud == UP
                    segments = segments[:iseg] + [Segment(corner.pos.col, seg.end)] + segments[iseg+1:]
                    seg = segments[iseg]

                elif corner.pos.col == seg.start:
                    if corner2.pos.col < seg.end:
                        assert corner.ud == UP and corner2.ud == DOWN
                        segments = segments[:iseg] + [Segment(corner2.pos.col, seg.end)] + segments[iseg+1:]
                        result += corner2.pos.col - seg.start
                        # print(f"          :  {result} <- += {corner2.pos.col - seg.start}")
                        seg = segments[iseg]
                    elif corner2.pos.col == seg.end:
                        assert corner.ud == UP and corner2.ud == UP
                        segments = segments[:iseg] + segments[iseg+1:]
                        result += seg.length
                        # print(f"          :  {result} <- += {seg.length}")
                        if iseg < len(segments):
                            seg = segments[iseg]

                elif seg.start < corner.pos.col < seg.end:
                    assert corner.ud == DOWN
                    if corner2.pos.col < seg.end:
                        assert corner2.ud == DOWN
                        segments = segments[:iseg] + [Segment(seg.start, corner.pos.col), Segment(corner2.pos.col, seg.end)] + segments[iseg+1:]
                        result += corner2.pos.col - corner.pos.col - 1
                        # print(f"          :  {result} <- += {corner2.pos.col - corner.pos.col - 1}")
                        iseg += 1
                        seg = segments[iseg]
                    elif corner2.pos.col == seg.end:
                        assert corner2.ud == UP
                        segments = segments[:iseg] + [Segment(seg.start, corner.pos.col)] + segments[iseg+1:]
                        result += seg.end - corner.pos.col
                        # print(f"          :  {result} <- += {seg.end - corner.pos.col}")
                        seg = segments[iseg]
                    else:
                        raise RuntimeError(f"No match for seg.end {seg.end}")

                elif corner.pos.col == seg.end:
                    assert corner.ud == UP
                    if iseg + 1 < len(segments) and corner2.ud == UP:
                        seg2 = segments[iseg+1]
                        assert corner2.pos.col == seg2.start
                        segments = segments[:iseg] + [Segment(seg.start, seg2.end)] + segments[iseg+2:]
                        seg = segments[iseg]
                    elif corner2.ud == DOWN:
                        segments = segments[:iseg] + [Segment(seg.start, corner2.pos.col)] + segments[iseg+1:]
                        seg = segments[iseg]

                elif corner.pos.col > seg.end:
                    segments.append(Segment(corner.pos.col, corner2.pos.col))
                    iseg += 1
                    seg = segments[iseg]

                else:
                    raise RuntimeError(f"No match for corner {corner}")

        return result

def pairs(vals: list[Any]) -> list[Tuple[Any, Any]]:
    assert len(vals) % 2 == 0
    if not vals:
        return []
    return zip(vals[::2], vals[1::2])


def parse_plan2(lines: list[str]) -> list[Tuple[str, int]]:
    result = []
    for line in lines:
        m = PLAN_RE.match(line)

        val = int(m.group(3), 16)
        direction = CODE_TO_DIR[val % 16]
        length = val // 16

        result.append((direction, length))
    return result

def parse_plan(lines: list[str]) -> list[Tuple[str, int]]:
    result = []
    for line in lines:
        m = PLAN_RE.match(line)
        direction, length, _ = m.groups()
        result.append((direction, int(length)))
    return result

def solve2(lines: Lines) -> int:
    """Solve the problem."""
    plan = parse_plan2(lines)
    board = Board2.from_plan(plan)
    # print(board)

    return board.volume()

def solve21(lines: Lines) -> int:
    """Solve the problem."""
    plan = parse_plan(lines)
    board = Board2.from_plan(plan)
    # print(board)

    return board.volume()

def solve(lines: Lines) -> int:
    """Solve the problem."""
    plan = parse_plan(lines)
    board = Board.from_plan(plan)
    # print(board)
    # print()

    for start in Pos(0, 0).all_neighbors():
        if board.fill(start):
            # print(board)
            # print()
            break

    return board.volume()


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
    assert result == 72821
    print("= " * 32)


# PART 2

def example2() -> None:
    """Run example for problem with input arguments."""
    print("EXAMPLE 2:")
    for text, expected in SAMPLE_CASES:
        lines = load_text(text)
        result = solve21(lines)
        print(f"'{text}' -> {result} (expected {expected})")
        assert result == expected

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
    assert result == 127844509405501
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    example2()
    part2(input_lines)
