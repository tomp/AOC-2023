#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 19
#
from typing import Sequence, Union, Optional, Any, Dict, List, Tuple
from pathlib import Path
from collections import defaultdict, deque, UserDict
from dataclasses import dataclass
from pprint import pprint
import math
import re

INPUTFILE = "input.txt"

SAMPLE_CASES = [
    (
        """
px{a<2006:qkq,m>2090:A,rfg}
pv{a>1716:R,A}
lnx{m>1548:A,A}
rfg{s<537:gd,x>2440:R,A}
qs{s>3448:A,lnx}
qkq{x<1416:A,crn}
crn{x>2662:A,R}
in{s<1351:px,qqz}
qqz{s>2770:qs,m<1801:hdj,R}
gd{a>3333:R,R}
hdj{m>838:A,pv}

{x=787,m=2655,a=1222,s=2876}
{x=1679,m=44,a=2067,s=496}
{x=2036,m=264,a=79,s=2244}
{x=2461,m=1339,a=466,s=291}
{x=2127,m=1623,a=2188,s=1013}
        """,
        19114
    ),
]

SAMPLE_CASES2 = [
    (
        """
px{a<2006:qkq,m>2090:A,rfg}
pv{a>1716:R,A}
lnx{m>1548:A,A}
rfg{s<537:gd,x>2440:R,A}
qs{s>3448:A,lnx}
qkq{x<1416:A,crn}
crn{x>2662:A,R}
in{s<1351:px,qqz}
qqz{s>2770:qs,m<1801:hdj,R}
gd{a>3333:R,R}
hdj{m>838:A,pv}

{x=787,m=2655,a=1222,s=2876}
{x=1679,m=44,a=2067,s=496}
{x=2036,m=264,a=79,s=2244}
{x=2461,m=1339,a=466,s=291}
{x=2127,m=1623,a=2188,s=1013}
        """,
        167409079868000
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

WORKFLOW_RE = re.compile(r"^([a-z]+){([^}]+)}")
RULE_RE = re.compile(r"^([a-z]+)([<>])(\d+):([a-z]+|[AR])$")
PART_RE = re.compile(r"^\{x=(\d+),m=(\d+),a=(\d+),s=(\d+)\}$")

ACCEPT, REJECT = "A", "R"
GT, LT = ">", "<"
THEN = ":"


@dataclass(frozen=True)
class Range:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


class PartClass(UserDict):
    def __init__(self, data: Optional[dict[str, Range]] = None):
        super().__init__(self)
        if data:
            self.data = {
                "x": data["x"],
                "m": data["m"],
                "a": data["a"],
                "s": data["s"],
            }
        else:
            self.data = {
                "x": Range(1,4000),
                "m": Range(1,4000),
                "a": Range(1,4000),
                "s": Range(1,4000),
            }

    @property
    def size(self) -> int:
        return (
            self.data['x'].size * self.data['m'].size * self.data['a'].size * self.data['s'].size
        )

    def copy(self) -> "PartClass":
        return PartClass(self.data)

    def split(self, rule: "Rule") -> Tuple[Optional["PartClass"], Optional["PartClass"]]:
        if not rule.op:
            return self, None

        if rule.op == GT:
            if self[rule.var].start > rule.val:
                return self, None
            if self[rule.var].end <= rule.val:
                return None, self
            if self.data[rule.var].end > rule.val:
                true_set, false_set = self.copy(), self.copy()
                true_set[rule.var] = Range(rule.val + 1, self[rule.var].end)
                false_set[rule.var] = Range(self[rule.var].start, rule.val)
                return true_set, false_set

        elif rule.op == LT:
            if self[rule.var].end < rule.val:
                return self, None
            if self[rule.var].start >= rule.val:
                return None, self
            if self[rule.var].end > rule.val:
                true_set, false_set = self.copy(), self.copy()
                true_set[rule.var] = Range(self[rule.var].start, rule.val-1)
                false_set[rule.var] = Range(rule.val, self[rule.var].end)
                return true_set, false_set

        raise RuntimeError(f"Invalid rule: '{rule}'")


class Rule:
    def __init__(self, var: str, op: str, val: str, goto: str):
        self.var = var
        self.op = op
        self.val = val
        self.goto = goto

        assert goto
        if op:
            assert op in (GT, LT) and var and val
            self.val = int(val)

    def __str__(self) -> str:
        if self.op:
            return f"if {self.var} {self.op} {self.val} then {self.goto}"
        if self.goto == ACCEPT:
            return "ACCEPT"
        if self.goto == REJECT:
            return "REJECT"
        return f"goto {self.goto}"

    def is_final(self) -> bool:
        return not self.op

    def apply(self, part: dict[str, int]) -> Optional[str]:
        if self.op == GT:
            if part[self.var] > self.val:
                return self.goto
        elif self.op == LT:
            if part[self.var] < self.val:
                return self.goto
        elif not self.op:
            return self.goto
        return None

    def classify(self, part: PartClass) -> Tuple[str, Optional[PartClass], Optional[PartClass]]:
        if_part, else_part = part.split(self)
        return self.goto, if_part, else_part



class Workflow:
    def __init__(self, rules: list[Rule]):
        self.rules = rules
        assert rules[-1].is_final()

    def __str__(self) -> str:
        return "(" + ", ".join([str(rule) for rule in self.rules]) + ")"

    def apply(self, part: dict[str, int]) -> str:
        for rule in self.rules:
            target = rule.apply(part)
            if target:
                return target
        raise RuntimeError("Workflow failed: " + ", ".join([str(rule) for rule in self.rules]))

    def classify(self, part: PartClass) -> list[Tuple[str, PartClass]]:
        result = []

        part_left = part
        for rule in self.rules:
            target, if_part, else_part = rule.classify(part_left)
            result.append((target, if_part))
            part_left = else_part
        assert part_left is None

        return result


def parse_input(lines) -> Tuple[dict[str, Workflow], list[dict[str, int]]]:
    sects = parse_sections(lines)
    assert len(sects) == 2

    parts = []
    for line in sects[1]:
        m = PART_RE.match(line)
        assert m
        part = {k: int(v) for k, v in zip("xmas", m.groups())}
        parts.append(part)

    workflows = {}
    for line in sects[0]:
        m = WORKFLOW_RE.match(line)
        assert m
        name, description = m.groups()
        rules = []
        for text in description.split(','):
            if THEN in text:
                m2 = RULE_RE.match(text)
                assert m2
                rules.append(Rule(*m2.groups()))
            else:
                rules.append(Rule("", "", "", text))
        workflows[name] = Workflow(rules)

    return workflows, parts


def solve2(lines: Lines) -> int:
    """Solve the problem."""
    workflows, _ = parse_input(lines)

    # for name, rules in sorted(workflows.items()):
    #     print(f"{name:4s} -> {str(rules)}")
    # print()

    accepted = []
    queue = deque([("in", PartClass())])
    while queue:
        name, part = queue.popleft()
        # print(f">>> apply {name} workflow to {str(part)}")
        for target, target_part in workflows[name].classify(part):
            if target == ACCEPT:
                # print(f"*** accept {str(target_part)}")
                accepted.append(target_part)
            elif target != REJECT:
                queue.append((target, target_part))

    result = sum((part.size for part in accepted))
    return result

def solve(lines: Lines) -> int:
    """Solve the problem."""
    workflows, parts = parse_input(lines)

    # for name, rules in sorted(workflows.items()):
    #     print(f"{name:4s} -> {str(rules)}")
    # print()

    result = 0
    for part in parts:
        # print(f"[x={part['x']},m={part['m']},a={part['a']},s={part['s']}]")
        name = "in"
        while name not in (ACCEPT, REJECT):
            new_name = workflows[name].apply(part)
            # print(f">>> {name} -> {new_name}")
            name = new_name
        if name == ACCEPT:
            result += sum(part.values())
            # print(f"{result} += {sum(part.values())}")

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
    assert result == 446935
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
    assert result == 141882534122898
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE, blank_lines=True)
    part1(input_lines)
    example2()
    part2(input_lines)
