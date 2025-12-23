"""
Run `cartpole_easy_demo` and perform some tests on its output. This should hopefully be
close enough to deterministic that we can test some things numerically across platforms, but
if these tests fail, check whether it might be a matter of allowable nondeterminism rather
than an actual bug.
"""

from contextlib import redirect_stdout
from io import StringIO

from nsai_experiments.general_az_1p.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import re

import pytest

from nsai_experiments.general_az_1p.cartpole.cartpole_easy_demo import main as cartpole_easy_demo_main


def parse_cartpole_output(raw_output):
    lines = raw_output.splitlines()
    preamble_lines = []
    iteration_blocks = []
    current_block = None

    for line in lines:
        if line.startswith("Training iteration"):
            current_block = [line]
            iteration_blocks.append(current_block)
            continue
        if current_block is None:
            preamble_lines.append(line)
            continue
        current_block.append(line)

    assert iteration_blocks, "Output missing training iterations."
    preamble = "\n".join(preamble_lines)
    iterations = ["\n".join(block) for block in iteration_blocks]
    return preamble, iterations

def match_one_group(pattern, text):
    match = re.search(pattern, text)
    assert match, "No match found."
    assert len(match.groups()) == 1, "Expected exactly one capture group."
    return match.group(1)

def run_game():
    cartpole_easy_demo_main()

@pytest.fixture(scope="module")
def game_output():
    """Run game once and capture output for all tests in this module."""
    buffer = StringIO()
    with redirect_stdout(buffer):
        run_game()
    output = buffer.getvalue()
    assert output, "run_game produced no stdout."
    return output

@pytest.fixture(scope="module")
def parsed_game_output(game_output):
    return parse_cartpole_output(game_output)

def test_preamble(parsed_game_output):
    """Test that device configuration appears in output."""
    assert "Neural network training will occur on device 'cpu'" in parsed_game_output[0]
    assert "Agent config" in parsed_game_output[0]
    assert "RNG seeds are fully specified" in parsed_game_output[0]

def test_iteration_count(parsed_game_output):
    """Test that expected number of training iterations occurred."""
    assert len(parsed_game_output[1]) == 3, f"Expected 3 training iterations, got {len(parsed_game_output[1])}."

def test_first_iteration(parsed_game_output):
    first_iteration = parsed_game_output[1][0]
    assert first_iteration.startswith("Training iteration 1")
    assert float(match_one_group(r"Total value: ([\d.]+)", first_iteration)) == 945.71
    assert int(match_one_group(r"Training on ([\d.]+) examples", first_iteration)) == 2637
    assert 0.21 <= float(match_one_group(r"Old network\+MCTS average reward: ([\d.]+)", first_iteration)) <= 0.23
    assert float(match_one_group(r"\(([\d.]+)% wins where ties are half wins\)", first_iteration)) >= 90.0

def test_last_iteration(parsed_game_output):
    last_iteration = parsed_game_output[1][-1]
    assert last_iteration.startswith("Training iteration 3")
    lengths = match_one_group(r"Training examples lengths: \[([^\]]*)\]", last_iteration)
    entries = [entry.strip() for entry in lengths.split(",") if entry.strip()]
    assert len(entries) == 2, f"Expected 2 training example lengths, got {len(entries)}."
    assert float(match_one_group(r"New network\+MCTS average reward: ([\d.]+)", last_iteration)) >= 0.75
