from .zg_gym import Tile
from nltk import CFG, PCFG, Nonterminal
import numpy as np

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

def print_ruleset(ruleset):
    print(" ".join(ruleset).replace("; ", ";\n"))
