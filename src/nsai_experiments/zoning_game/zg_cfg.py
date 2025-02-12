from collections import namedtuple

from nltk.tokenize import wordpunct_tokenize
from nltk.parse.recursivedescent import RecursiveDescentParser
from nltk import CFG, PCFG, Nonterminal
import numpy as np

from .zg_gym import Tile

MAX_NUM = 6  # TODO do not hardcode
ZONING_GAME_GRAMMAR_STRING = f"""
    Policy -> Rule ";" Policy [0.8] | "" [0.2]
    Rule -> Subject "must" Constraint [1.0]
    Constraint -> "(" Constraint "or" Constraint ")" [0.15]
    Constraint -> "(" Constraint "and" Constraint ")" [0.15]
    Constraint -> "(" "not" Constraint ")" [0.25]
    Constraint -> DistanceConstraint [0.15]
    Constraint -> ClusterCountConstraint [0.15]
    Constraint -> ClusterSizeConstraint [0.15]
    DistanceConstraint -> "be_within" Number "tiles_of" Object [1.0]
    ClusterCountConstraint -> "form_fewer_than" Number "separate_clusters" [1.0]
    ClusterSizeConstraint -> "form_cluster_with_fewer_than" Number "tiles" [1.0]
    Subject -> Tile [1.0]
    Object -> Tile [0.5] | Location [0.5]
    Tile -> {" | ".join([f"\"{x.name}\" [{1/(len(Tile)-1):.4f}]" for x in Tile if x is not Tile.EMPTY])}
    Location -> "board_center" [0.2] | "board_edge" [0.5] | "board_corner" [0.3]
    Number -> {" | ".join([f"\"{x}\" [{1/(MAX_NUM-1):.4f}]" for x in range(1, MAX_NUM)])}
                                     """
ZONING_GAME_GRAMMAR = PCFG.fromstring(ZONING_GAME_GRAMMAR_STRING)

RuleNT = namedtuple("Rule", ["subject", "constraint"])
OrNT = namedtuple("Or", ["sub1", "sub2"])
AndNT = namedtuple("And", ["sub1", "sub2"])
NotNT = namedtuple("Not", ["sub"])
DistanceConstraintNT = namedtuple("DistanceConstraint", ["distance", "object"])
ClusterCountConstraintNT = namedtuple("ClusterCountConstraint", ["count"])
ClusterSizeConstraintNT = namedtuple("ClusterSizeConstraint", ["size"])

def generate_one_probabilistic(pcfg: PCFG, current_nonterminal = None, seed = None, rng = None):
    assert seed is None or rng is None
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
        seed = None
    if current_nonterminal is None: current_nonterminal = pcfg.start()
    current_prods = list(pcfg.productions(lhs = current_nonterminal))
    selected_prod = (np.random if rng is None else rng).choice(current_prods, p = [prod.prob() for prod in current_prods])
    result = []
    for fragment in selected_prod.rhs():
        result += generate_one_probabilistic(pcfg, fragment, seed=seed, rng=rng) if isinstance(fragment, Nonterminal) else [fragment]
    return result

def format_ruleset(ruleset):
    return " ".join(ruleset).replace("; ", ";\n")

ZONING_GAME_PARSER = RecursiveDescentParser(ZONING_GAME_GRAMMAR)
def parse_formatted_ruleset(formatted_ruleset):
    root, = ZONING_GAME_PARSER.parse(wordpunct_tokenize(formatted_ruleset)+[""])
    return root

def _process_subject_object(subject_object_ast):
    so_type = subject_object_ast.label()
    so_id, = subject_object_ast
    match so_type:
        case "Tile":
            return Tile[so_id]
        case "Location":
            return so_id
        case "Number":
            return int(so_id)
        case _:
            raise ValueError()

def _process_base_constraint(base_constraint_ast):
    match base_constraint_ast.label():
        case "DistanceConstraint":
            _, dist, _, obj = base_constraint_ast
            dist, = dist
            obj, = obj
            return DistanceConstraintNT(int(dist), _process_subject_object(obj))
        case "ClusterCountConstraint":
            _, count, _ = base_constraint_ast
            count, = count
            return ClusterCountConstraintNT(int(count))
        case "ClusterSizeConstraint":
            _, size, _ = base_constraint_ast
            size, = size
            return ClusterSizeConstraintNT(int(size))
        case _:
            raise ValueError()

def _process_constraint(constraint_ast):
    match len(constraint_ast):
        case 1:
            return _process_base_constraint(constraint_ast[0])
        case 4:
            _, not_str, sub, _ = constraint_ast
            assert not_str == "not"
            return NotNT(_process_constraint(sub))
        case 5:
            _, sub1, op_str, sub2, _ = constraint_ast
            match op_str:
                case "and":
                    op = AndNT
                case "or":
                    op = OrNT
                case _:
                    raise ValueError()
            return op(_process_constraint(sub1), _process_constraint(sub2))
        case _:
            raise ValueError()

def _process_rule(rule_ast):
    subject, _, constraint = rule_ast
    subject, = subject
    subject = _process_subject_object(subject)
    constraint = _process_constraint(constraint)
    return RuleNT(subject, constraint)

def _extract_rule_list(parsed_ruleset):
    "A parser of sorts that turns a ruleset AST into an intermediate representation that is easier to work with"
    if len(parsed_ruleset) == 3:  # Handles `Policy -> Rule ";" Policy`
        rule, _, rest = parsed_ruleset
        return [_process_rule(rule)] + _extract_rule_list(rest)
    else:  # Handles `Policy -> ""`
        assert list(parsed_ruleset) == [""]
        return []

def _interpret_one_constraint(constraint, tile_grid, my_row, my_col):
    args = (tile_grid, my_row, my_col)
    return True  # TODO placeholder

def _interpret_indiv(rule_list, tile_grid, my_row, my_col):
    "Like `interpret_indiv` but operates on a rule list rather than a raw AST"
    my_tile = Tile(tile_grid[my_row, my_col])
    my_rules = filter(lambda rule: rule.subject == my_tile, rule_list)
    return all([_interpret_one_constraint(rule.constraint, tile_grid, my_row, my_col) for rule in my_rules])

def interpret_indiv(parsed_ruleset, tile_grid, my_row, my_col):
    """
    Check the given tile of the tile grid against the parsed ruleset AST and return whether
    it complies. If we view sentences from the zoning game language as programs that
    describe which tile configurations are allowed, this is an interpreter of such programs.
    """
    rule_list = _extract_rule_list(parsed_ruleset)
    return _interpret_indiv(rule_list, tile_grid, my_row, my_col)

def interpret_grid(parsed_ruleset, tile_grid):
    "Like `interpret_indiv` but returns a Boolean array of per-tile results across an entire tile grid."
    pass  # TODO placeholder
