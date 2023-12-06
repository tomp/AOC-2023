#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 6
#
import sys
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
        Time:      7  15   30
        Distance:  9  40  200
        """,
        288
    ),
]

SAMPLE_CASES2 = [
    (
        """
        Time:      7  15   30
        Distance:  9  40  200
        """,
        71503
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

@dataclass
class Race:
    time: int
    dist: int

    def dist_for_wait(self, wait: int) -> int:
        if wait == 0 or wait >= self.time:
            return 0
        return (self.time - wait) * wait

    def best_dist(self) -> float:
        return (self.time * self.time) / 4

    def beat_record(self) -> int:
        """How many ways are there to beat the record? """
        result = 0

        disc = 0.5 * math.sqrt((self.time * self.time) - (4 * self.dist))
        time0 = int(0.5 * self.time - disc)
        time1 = int(0.5 * self.time + disc)

        result = time1 - time0 + 1
        if self.dist_for_wait(time0) <= self.dist:
            result -= 1
        if self.dist_for_wait(time1) == self.dist:
            result -= 1
        return result

    def beat_record_slow(self) -> int:
        """How many ways are there to beat the record? """
        result = 0
        for x in range(self.time):
            dist = self.dist_for_wait(x)
            if dist > self.dist:
                result += 1
        return result


def parse_input(lines) -> Sequence[Race]:
    times, dists = [], []
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("Time:"):
            times = list(map(int, line.strip().split()[1:]))
        elif line.startswith("Distance:"):
            dists = list(map(int, line.strip().split()[1:]))
        else:
            raise ValueError(f"Unexpected line: '{line.strip()}'")

    assert len(times) == len(dists) and len(times) > 0
    return [Race(t, d) for t, d in zip(times, dists)]


def parse_input2(lines) -> Race:
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("Time:"):
            times = int("".join(line.strip().split()[1:]))
        elif line.startswith("Distance:"):
            dists = int("".join(line.strip().split()[1:]))
        else:
            raise ValueError(f"Unexpected line: '{line.strip()}'")

    return Race(times, dists)

def solve2(lines: Lines) -> int:
    """Solve the problem."""
    race = parse_input2(lines)
    return race.beat_record()

def solve(lines: Lines) -> int:
    """Solve the problem."""
    races = parse_input(lines)
    result = 1
    for race in races:
        result *= race.beat_record()
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
    assert result == 800280
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
    assert result == 45128024
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    example2()
    part2(input_lines)
