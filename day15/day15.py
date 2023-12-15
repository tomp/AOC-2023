#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 15
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
        rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7
        """,
        1320
    ),
]

SAMPLE_CASES2 = [
    (
        """
        rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7
        """,
        145
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

WORD_RE = re.compile(r"([a-z]+)([=-])(\d+)?$")

@dataclass
class Lens:
    label: str
    focal: int

class Box:
    def __init__(self, items: Optional[list[Lens]] = None):
        self.labels = []
        self.focals = []
        if items:
            for item in items:
                self.labels.append(lens.label)
                self.focals.append(lens.focal)

    def __str__(self):
        return ", ".join(
            [f"({label} {focal})" for label, focal in zip(self.labels, self.focals)]
        )

    def power(self) -> int:
        result = 0
        for idx, focal in enumerate(self.focals):
            result += (idx + 1) * focal
        return result

    @property
    def lenses(self):
        return [Lens(label, focal) for label, focal in zip(self.labels, self.focals)]

    def remove(self, label):
        try:
            idx = self.labels.index(label)
            self.labels = self.labels[:idx] + self.labels[idx+1:]
            self.focals = self.focals[:idx] + self.focals[idx+1:]
        except ValueError:
            pass
        return self

    def add(self, label: str, focal: int):
        if label in self.labels:
            idx = self.labels.index(label)
            self.focals[idx] = focal
        else:
            self.labels.append(label)
            self.focals.append(focal)
        return self


def hash_algo(text: str) -> int:
    result = 0
    for ch in text:
        result = ((result + ord(ch)) * 17) % 256
    return result


ADD, SUB = "=", "-"

def parse_input(lines):
    boxes = defaultdict(Box)
    for word in lines[0].strip().split(","):
        m = WORD_RE.match(word)
        label = m.group(1)
        op = m.group(2)
        focal = m.group(3)
        boxnum = hash_algo(label)
        if op == ADD:
            boxes[boxnum].add(label, int(focal))
        elif op == SUB:
            boxes[boxnum].remove(label)
    return boxes


def solve2(lines: Lines) -> int:
    """Solve the problem."""
    result = 0

    boxes = parse_input(lines)

    for num, item in sorted(boxes.items()):
        # print(f"box {num}:  {item.power():3d} {str(item)}")
        result += (num + 1) * item.power()

    return result

def solve(lines: Lines) -> int:
    """Solve the problem."""
    result = 0
    for word in lines[0].strip().split(","):
        result += hash_algo(word)
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
    assert result == 518107
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
    assert result == 303404
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    example2()
    part2(input_lines)
