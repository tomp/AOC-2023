#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 9
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
        0 3 6 9 12 15
        1 3 6 10 15 21
        10 13 16 21 30 45
        """,
        114
    ),
]

SAMPLE_CASES2 = [
    (
        """
        0 3 6 9 12 15
        1 3 6 10 15 21
        10 13 16 21 30 45
        """,
        2
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

def parse_input(lines):
    result = []
    for line in lines:
        if line.strip():
            result.append(list(map(int, line.split())))
    return result

def solve2(lines: Lines) -> int:
    """Solve the problem."""
    history = parse_input(lines)
    result = 0
    for values in history:
        diffs = [values]
        while any([val != 0 for val in diffs[-1]]):
            diffs.append([t1 - t0 for t0, t1 in zip(diffs[-1][:-1], diffs[-1][1:])])
        prev_value = 0
        for vals in diffs[::-1]:
            prev_value = vals[0] - prev_value
        result += prev_value
    return result

def solve(lines: Lines) -> int:
    """Solve the problem."""
    history = parse_input(lines)
    result = 0
    for values in history:
        diffs = [values]
        while any([val != 0 for val in diffs[-1]]):
            diffs.append([t1 - t0 for t0, t1 in zip(diffs[-1][:-1], diffs[-1][1:])])
        next_value = sum([row[-1] for row in diffs])
        result += next_value
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
    assert result == 1916822650
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
    assert result == 966
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    example2()
    part2(input_lines)
