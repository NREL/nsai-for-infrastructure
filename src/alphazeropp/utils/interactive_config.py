"""
Reusable interactive config editor for CLI scripts.

Provides a numbered-parameter display and edit loop that works with any
config object (argparse.Namespace, dataclass, nested dicts, etc.).
Each script supplies a `build_sections` callable that returns the current
parameter layout; the generic code handles display, prompting, parsing,
and choice menus.

Usage:
    from alphazeropp.utils.interactive_config import (
        build_param_list, interactive_edit, attr_setter,
    )

    def _build_sections(args):
        params = [
            ("n_sites", args.n_sites, attr_setter(args, "n_sites"), "Number of bits", None),
            ("metric",  args.metric,  attr_setter(args, "metric"),  "Eval metric",    ["a", "b"]),
        ]
        all_params = build_param_list(params)
        return [("Section", all_params)]

    interactive_edit("My Config", lambda: _build_sections(args))
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


# ---------------------------------------------------------------------------
# Type-aware value parser
# ---------------------------------------------------------------------------

def parse_value(raw: str, current_value: Any) -> Any:
    """Parse a string input to the same type as *current_value*.

    Handles bool, None, int, float, and str.
    """
    t = type(current_value)
    if t is bool:
        return raw.strip().lower() in ("true", "1", "yes")
    if current_value is None:
        return None if raw.strip().lower() in ("none", "") else raw.strip()
    return t(raw)


# ---------------------------------------------------------------------------
# Param list builder
# ---------------------------------------------------------------------------

# A "param spec" is (label, value, setter, description, choices).
# A "param tuple" adds an auto-assigned number at index 0.
ParamSpec = tuple[str, Any, Callable[[Any], None], str, list | None]
ParamTuple = tuple[int, str, Any, Callable[[Any], None], str, list | None]


def build_param_list(specs: Sequence[ParamSpec]) -> list[ParamTuple]:
    """Auto-number a flat list of param specs into param tuples.

    Numbers start at 1 and increment sequentially.
    """
    return [
        (i + 1, label, value, setter, desc, choices)
        for i, (label, value, setter, desc, choices) in enumerate(specs)
    ]


# ---------------------------------------------------------------------------
# Setter helpers
# ---------------------------------------------------------------------------

def attr_setter(obj: Any, key: str) -> Callable[[Any], None]:
    """Return a setter that does ``setattr(obj, key, val)``."""
    return lambda val, _o=obj, _k=key: setattr(_o, _k, val)


def dict_setter(d: dict, key: str) -> Callable[[Any], None]:
    """Return a setter that does ``d[key] = val``."""
    return lambda val, _d=d, _k=key: _d.__setitem__(_k, val)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

Section = tuple[str, list[ParamTuple]]


def display_sections(title: str, sections: list[Section]) -> dict[int, ParamTuple]:
    """Print a numbered parameter table grouped by sections.

    Returns a ``{number: param_tuple}`` map for lookup by the edit loop.
    """
    print(f"\n=== {title} ===\n")
    param_map: dict[int, ParamTuple] = {}
    for section_name, section_params in sections:
        if not section_params:
            continue
        print(f"  {section_name}:")
        for num, label, value, setter, desc, choices in section_params:
            line = f"    {num:>2}) {label:<22} = {str(value):<10}"
            if desc:
                line += f"  # {desc}"
            print(line)
            param_map[num] = (num, label, value, setter, desc, choices)
        print()
    return param_map


# ---------------------------------------------------------------------------
# Edit loop
# ---------------------------------------------------------------------------

def interactive_edit(
    title: str,
    build_sections: Callable[[], list[Section]],
) -> None:
    """Interactive editing loop.

    *build_sections* is called at the start of each iteration to get the
    current parameter layout (so conditional params refresh correctly).

    The loop displays the config, prompts the user for a parameter number
    to edit, applies the change, and repeats until the user types ``run``,
    ``start``, or presses Enter.
    """
    while True:
        sections = build_sections()
        param_map = display_sections(title, sections)

        choice = input("  Enter number to edit (or 'run' to start): ").strip()
        if choice.lower() in ("run", "start", ""):
            break

        try:
            num = int(choice)
        except ValueError:
            print(f"  Invalid input: '{choice}'. Enter a number or 'run'.")
            continue

        if num not in param_map:
            print(f"  No parameter with number {num}.")
            continue

        _, label, current, setter, _, choices = param_map[num]

        if choices is not None:
            print(f"  {label} options:")
            for i, c in enumerate(choices):
                print(f"    {i}) {c}")
            raw = input(f"  Select [0-{len(choices)-1}]: ").strip()
            if raw == "":
                continue
            try:
                idx = int(raw)
                if 0 <= idx < len(choices):
                    setter(choices[idx])
                    print(f"  Updated: {label} = {choices[idx]}")
                else:
                    print(f"  Invalid selection: {idx}")
            except ValueError:
                print(f"  Invalid input: '{raw}'. Enter a number.")
        else:
            raw = input(f"  New value for {label} ({type(current).__name__}) [{current}]: ").strip()
            if raw == "":
                continue
            try:
                new_val = parse_value(raw, current)
                setter(new_val)
                print(f"  Updated: {label} = {new_val}")
            except (ValueError, TypeError) as e:
                print(f"  Invalid value: {e}")
