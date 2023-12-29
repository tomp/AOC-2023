#!/usr/bin/env python3
#
#  Advent of Code 2023 - Day 20
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
broadcaster -> a, b, c
%a -> b
%b -> c
%c -> inv
&inv -> a
        """,
        32000000
    ),
    (
        """
broadcaster -> a
%a -> inv, con
&inv -> b
%b -> con
&con -> output
        """,
        11687500
    ),
]

SAMPLE_CASES2 = SAMPLE_CASES


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

FLIP, CONJ = "%", "&"
BUTTON, BROAD = "button", "broadcaster"
ARROW = "->"
ON, OFF = True, False
HIGH, LOW = "high", "low"


@dataclass
class Signal:
    origin: str
    dest: str
    level: int


class Module:
    def __init__(self, name: str, typ: str, outputs: list[str]):
        self.name = name
        self.typ = typ
        self.outputs = outputs

        self.state = OFF
        self.inputs = {}

        assert typ in (FLIP, CONJ) or name == BROAD or len(outputs) == 0

    def __str__(self) -> str:
        return f"{self.typ}{self.name} -> " + ", ".join(self.outputs)

    def initial_state(self) -> bool:
        if self.typ == FLIP:
            return self.state == OFF

        if self.typ == CONJ:
            return all((val == LOW for val in self.inputs.values()))

        return True

    def connect(self, origin):
        self.inputs[origin] = LOW

    def receive(self, signal: "Signal", m: "Machine") -> list["Signal"]:

        if self.typ == FLIP:
            if signal.level == LOW:
                self.state = not self.state
                if self.state:
                    return [Signal(self.name, dest, HIGH) for dest in self.outputs]
                return [Signal(self.name, dest, LOW) for dest in self.outputs]
            return []

        if self.typ == CONJ:
            self.inputs[signal.origin] = signal.level
            if all((val == HIGH for val in self.inputs.values())):
                return [Signal(self.name, dest, LOW) for dest in self.outputs]
            return [Signal(self.name, dest, HIGH) for dest in self.outputs]

        assert self.name == BROAD or len(self.outputs) == 0

        if self.name == "rx" and signal.level == LOW:
            print(f"[{m.pushed}:{m.signals}]  {signal.origin} -{signal.level}-> {signal.dest}")
            raise RuntimeError("DONE")

        return [Signal(self.name, dest, signal.level) for dest in self.outputs]


class Machine:
    def __init__(self, modules: dict[str, Module]):
        self.modules = modules
        self.queue = deque()
        self.pushed = 0
        self.ll_input = {}
        self.signals = 0
        self.high = 0
        self.low = 0

        assert BROAD in modules

        names = list(self.modules.keys())
        for name in names:
            for dest in self.modules[name].outputs:
                if dest not in self.modules:
                    self.modules[dest] = Module(dest, "", [])
                self.modules[dest].connect(name)

        if "ll" in self.modules:
            for name in self.modules["ll"].inputs.keys():
                self.ll_input[name] = 0

    def __str__(self) -> str:
        lines = []
        for _, module in sorted(self.modules.items()):
            lines.append(str(module))
        return "\n".join(lines)

    def initial_state(self) -> bool:
        return all((m.initial_state() for m in self.modules.values()))

    def button(self):
        self.pushed += 1
        self.signals = 0
        self.high = 0
        self.low = 0
        self.queue.append(Signal(BUTTON, BROAD, LOW))

    def propagate(self) -> int:
        if self.queue:
            signal = self.queue.popleft()
            self.signals += 1
            # print(f"{signal.origin} -{signal.level}-> {signal.dest}")
            if signal.level == HIGH:
                self.high +=1
            else:
                self.low += 1

            if signal.dest == "ll" and signal.level == HIGH:
                self.ll_input[signal.origin] = self.pushed

            for new_signal in self.modules[signal.dest].receive(signal, self):
                self.queue.append(new_signal)
        return len(self.queue)

    def backlog(self) -> int:
        return len(self.queue)

    def done(self) -> bool:
        return all(self.ll_input.values())

    def rx_pushes(self) -> int:
        return math.lcm(*self.ll_input.values())

    def stats(self) -> Tuple[int, int]:
        return self.low, self.high



MODULE_RE = re.compile(r"^([%&]?)([a-z]+) -> (.*)")


def parse_input(lines) -> Machine:
    modules = {}
    for line in lines:
        m = MODULE_RE.match(line)
        assert m, f"Unable to parse '{line}'"
        typ, name, rest = m.groups()
        outputs = rest.replace(" ", "").split(",")
        assert typ or name == BROAD
        modules[name] = Module(name, typ, outputs)
    return Machine(modules)


def solve2(lines: Lines) -> int:
    """Solve the problem."""
    machine = parse_input(lines)

    while not machine.done():
        machine.button()
        while machine.backlog():
            machine.propagate()
    result = machine.rx_pushes()
    return result


def solve(lines: Lines) -> int:
    """Solve the problem."""
    machine = parse_input(lines)
    print(machine)
    print()

    pushed, total_pushes = 0, 1000
    total_low, total_high = 0, 0
    while pushed < total_pushes:
        # print("\npush button...")
        machine.button()
        pushed += 1
        while machine.backlog():
            machine.propagate()

        low, high = machine.stats()
        total_low += low
        total_high += high
        if machine.initial_state():
            print(f"\nmachine returned to initial state after {pushed} button pushes")
            print(f"total pulses:  HIGH {total_high}  LOW {total_low}")
            cycles = total_pushes // pushed
            total_pushes -= cycles * pushed
            print(f"\nRun {cycles} cycles of {pushed} button pushes, plus {total_pushes} additional")
            total_high *= cycles
            total_low *= cycles

    print(f"total pulses:  HIGH {total_high}  LOW {total_low}")

    return total_high * total_low


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
    assert result == 743090292
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
    assert result == 241528184647003
    print("= " * 32)


if __name__ == "__main__":
    example1()
    input_lines = load_input(INPUTFILE)
    part1(input_lines)
    part2(input_lines)
