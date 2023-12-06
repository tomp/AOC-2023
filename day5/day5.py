#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 5
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
seeds: 79 14 55 13

seed-to-soil map:
50 98 2
52 50 48

soil-to-fertilizer map:
0 15 37
37 52 2
39 0 15

fertilizer-to-water map:
49 53 8
0 11 42
42 0 7
57 7 4

water-to-light map:
88 18 7
18 25 70

light-to-temperature map:
45 77 23
81 45 19
68 64 13

temperature-to-humidity map:
0 69 1
1 0 69

humidity-to-location map:
60 56 37
56 93 4
        """,
        35
    ),
]

SAMPLE_CASES2 = [
    (
        """
seeds: 79 14 55 13

seed-to-soil map:
50 98 2
52 50 48

soil-to-fertilizer map:
0 15 37
37 52 2
39 0 15

fertilizer-to-water map:
49 53 8
0 11 42
42 0 7
57 7 4

water-to-light map:
88 18 7
18 25 70

light-to-temperature map:
45 77 23
81 45 19
68 64 13

temperature-to-humidity map:
0 69 1
1 0 69

humidity-to-location map:
60 56 37
56 93 4
        """,
        46
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
        print(f">>> '{line}'")
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

@dataclass(frozen=True, order= True)
class Range:
    start: int
    end: int
    shift: int


MAX_RUN_LENGTH = 10000000000

class AgMap:
    """An AgMap instance maps src to dest values for one discrete range of values.
    """
    def __init__(self, mappings: list[Tuple[int, int, int]]):
        self.ranges = []
        for dest, src, size in mappings:
            self.ranges.append(Range(src, src + size, dest - src))
        self.ranges.sort()

        for i in range(len(self.ranges)-1):
            assert self.ranges[i].end <= self.ranges[i+1].start

        self.start = self.ranges[0].start
        self.end = self.ranges[-1].end
        self.last_range = self.ranges[0]

    def run_length(self) -> int:
        return last_range.end

    def src_to_dest(self, val: int) -> int:
        if self.last_range.start <= val < self.last_range.end:
            self.run_length = self.last_range.end - val
            result = val + self.last_range.shift

        elif val < self.start:
            self.run_length = self.start - val
            result = val

        elif val >= self.end:
            self.run_length = MAX_RUN_LENGTH
            result = val

        else:
            result = val
            for range in self.ranges:
                if range.start <= val < range.end:
                    result = val + range.shift
                    self.run_length = range.end - val
                    self.last_range = range
                    break

        return result


SEEDS_RE = re.compile(r"seeds:\s+(\d.*\d)\s*$")
MAP_RE = re.compile(r"([a-z]+)-to-([a-z]+) map:$")

def parse_input(lines):
    m = SEEDS_RE.match(lines[0])
    seeds = list(map(int, m.group(1).split()))

    maps = []
    regions = None
    for line in lines[1:]:
        if not line.strip():
            continue

        m = MAP_RE.match(line)
        if m:
            src_name, dest_name = m.groups()
            if regions is not None:
                maps.append(AgMap(regions))
            regions = []
        elif regions is not None:
            dest, src, size = list(map(int, line.split()))
            regions.append((dest, src, size))
    if regions:
        maps.append(AgMap(regions))

    return seeds, maps


def solve2(lines: Lines) -> int:
    """Solve the problem."""
    seeds, maps = parse_input(lines)
    result = -1

    seed_ranges = []
    while seeds:
        (start, count), seeds = seeds[:2], seeds[2:]
        seed_ranges.append((start, start+count))
    seed_ranges.sort()

    # total = sum([v[1] - v[0] for v in seed_ranges])
    # print(f"Total seeds to consider: {total}")

    for start, end in seed_ranges:
        seed = start
        while seed < end:
            val = seed
            runlen = end - start
            for agmap in maps:
                val = agmap.src_to_dest(val)
                runlen = min(runlen, agmap.run_length)
            if result < 0 or val < result:
                result = val
            seed += runlen

    return result


def solve(lines: Lines) -> int:
    """Solve the problem."""
    seeds, maps = parse_input(lines)
    result = -1

    for seed in seeds:
        val = seed
        print(f"\nseed: {seed}")
        for agmap in maps:
            val = agmap.src_to_dest(val)
            print(f"--> {val}")
        loc = val
        if result < 0 or loc < result:
            result = loc
            print(f"*** {result}")

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
    assert result == 484023871
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
    assert result == 46294175
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    example2()
    part2(input_lines)
