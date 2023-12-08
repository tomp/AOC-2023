#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 8
#
from typing import Sequence, Union, Optional, Any, Dict, List, Tuple
from pathlib import Path
from collections import defaultdict
from itertools import cycle
from dataclasses import dataclass
from pprint import pprint
import math
import re

INPUTFILE = "input.txt"

SAMPLE_CASES = [
    (
        """
        RL

        AAA = (BBB, CCC)
        BBB = (DDD, EEE)
        CCC = (ZZZ, GGG)
        DDD = (DDD, DDD)
        EEE = (EEE, EEE)
        GGG = (GGG, GGG)
        ZZZ = (ZZZ, ZZZ)
        """,
        2
    ),
    (
        """
        LLR

        AAA = (BBB, BBB)
        BBB = (AAA, ZZZ)
        ZZZ = (ZZZ, ZZZ)
        """,
        6
    ),
]

SAMPLE_CASES2 = [
    (
        """
        LR

        11A = (11B, XXX)
        11B = (XXX, 11Z)
        11Z = (11B, XXX)
        22A = (22B, XXX)
        22B = (22C, 22C)
        22C = (22Z, 22Z)
        22Z = (22B, 22B)
        XXX = (XXX, XXX)
        """,
        6
    ),
]


Lines = Sequence[str]
Sections = Sequence[Lines]

# Utility functions

def load_input(infile: str, strip=True, blank_lines=False) -> Lines:
    return load_text(Path(infile).read_text(), strip=strip, blank_lines=blank_lines)

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

ELEMENT_RE = re.compile(r"([A-Z\d]{3}) = \(([A-Z\d]{3}), ([A-Z\d]{3})\)\s*$")

def parse_input(lines):
    sects = parse_sections(lines)
    pprint(sects)

    elements = {}
    for line in sects[1]:
        m = ELEMENT_RE.match(line)
        elem, left, right = m.groups()
        elements[elem] = {"L": left, "R": right}
    dirs = list(sects[0][0].strip())
    return dirs, elements


def solve2(lines: Lines) -> int:
    """Solve the problem."""
    dirs, elements = parse_input(lines)
    elems = [elem for elem in elements.keys() if elem.endswith('A')]

    steps = 0
    found = defaultdict(int)
    gap = defaultdict(int)
    for dir in cycle(dirs):
        steps += 1
        elems = [elements[elem][dir] for elem in elems]

        # Look for cycle periods in finding the *Z words...
        if any([elem.endswith('Z') for elem in elems]):
            report = []
            for elem in elems:
                if elem.endswith('Z'):
                    if found[elem]:
                        gap[elem] = steps - found[elem]
                        report.append(f"{elem:3s}: {gap[elem]:5d} ")
                    found[elem] = steps
                else:
                    report.append("---:       ")
            print(steps, ", ".join(report))

        # Once we have a cycle length for each element, stop and
        # calculate how long it would take for them all to align
        if len(gap) == len(elems) and all(gap.values()):
            print()
            start = {elem:(found[elem] % period) for elem, period in gap.items()}
            for i, elem in enumerate(gap.keys()):
                print(f"steps = {start[elem]:6d} + n{i} * {gap[elem]:6d}")
            print()
            assert all([val == 0 for val in start.values()])

            # if all the cycles start at 0, we just need to know the
            # least common multiple of all the periods.
            steps = math.lcm(*gap.values())
            break

    return steps

def solve(lines: Lines) -> int:
    """Solve the problem."""
    dirs, elements = parse_input(lines)
    result = 0
    elem = "AAA"
    for dir in cycle(dirs):
        result += 1
        elem = elements[elem][dir]
        if elem == "ZZZ":
            break
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
    assert result == 11567
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
    assert result == 9858474970153
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE, blank_lines=True)
    part1(input_lines)
    example2()
    part2(input_lines)
