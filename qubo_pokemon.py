# qubo_pokemon.py
import math
import itertools
import pandas as pd
import numpy as np
import dimod
from neal import SimulatedAnnealingSampler

#parameters:
POKEMON_CSV = "data/pokemon.csv"        # expected cols: name,type1,type2 (type2 optional)
TYPE_CHART_CSV = "data/type_chart.csv"  # expected as multiplier table
MAX_PER_COMBO = 6     # each type combo can appear up to this many times on team
TEAM_SIZE = 6
P1 = 50.0  # team size penalty (tune if needed)
P2 = 40.0  # weakness penalty per attack type (tune)
P3 = 40.0  # resistance penalty per attack type (tune)

#Load data:
def load_pokemon():
    df = pd.read_csv(POKEMON_CSV)
    return df

def load_type_chart():
    tc = pd.read_csv(TYPE_CHART_CSV, index_col=0)
    tc = tc.transpose()
    return tc

# Build type combos and maps:
def canonical_combo(t1, t2):
    if pd.isna(t2) or t2 == "" or t2 is None:
        return t1
    # keep consistent ordering (alphabetic) so combos like Fire/Flying are canonical
    ordered = "/".join(sorted([t1, t2]))
    return ordered

def build_type_combos(pokemon_df):
    combos = pokemon_df.apply(lambda r: canonical_combo(r['type1'], r.get('type2', None)), axis=1)
    combos_unique = sorted(combos.unique())
    return combos_unique

# Build binary variables for each type-combo to represent count c_i in [0, MAX_PER_COMBO]
# We use binary expansion: bits with weights 1,2,4, ...
def bits_needed(max_value):
    return math.ceil(math.log2(max_value + 1))

def make_combo_bitnames(combos, max_per_combo):
    nbits = bits_needed(max_per_combo)
    combo_bits = {}
    weights = [2**i for i in range(nbits)]
    for c in combos:
        names = [f"x_{c}_b{j}" for j in range(nbits)]
        combo_bits[c] = {'bits': names, 'weights': weights}
    return combo_bits

# Generic helper to add penalty B*(sum_i a_i*b_i - rhs)^2 to bqm
# where b_i are binary variables, a_i are integer/float coefficients (weights of bits in that sum)
# This will expand to linear and quadratic terms and add them to the dimod BQM (binary variables)
def add_quadratic_equality_bqm(bqm, linear_coeffs, quadratic_coeffs, constant, strength):
    # utility that directly updates bqm from provided linear and quadratic dicts
    for v, coeff in linear_coeffs.items():
        bqm.add_variable(v, strength * coeff)
    for (u, v), coeff in quadratic_coeffs.items():
        bqm.add_interaction(u, v, strength * coeff)
    # add constant term as offset
    bqm.offset += strength * constant

def add_penalty_sum_equals(bqm, var_weight_map, rhs, penalty):
    """
    var_weight_map: dict mapping variable_name -> coefficient in the sum (a_i)
    rhs: the RHS scalar
    penalty: multiplicative penalty
    Adds penalty * (sum_i (a_i * var_i) - rhs)^2
    """
    # Expand (Σ a_i x_i - rhs)^2 = Σ a_i^2 x_i + 2Σ_{i<j} a_i a_j x_i x_j - 2 rhs Σ a_i x_i + rhs^2
    linear = {}
    quadratic = {}
    const = rhs * rhs

    # linear terms
    for v, a in var_weight_map.items():
        linear[v] = linear.get(v, 0.0) + (a * a)

    # quadratic terms
    items = list(var_weight_map.items())
    for i in range(len(items)):
        vi, ai = items[i]
        for j in range(i+1, len(items)):
            vj, aj = items[j]
            quadratic[(vi, vj)] = quadratic.get((vi, vj), 0.0) + 2.0 * ai * aj

    # cross linear with rhs
    for v, a in var_weight_map.items():
        linear[v] = linear.get(v, 0.0) + (-2.0 * rhs * a)

    # now add scaled by penalty
    add_quadratic_equality_bqm(bqm, linear, quadratic, const, penalty)

# Build the BQM
def build_bqm(combos, combo_bits, type_chart, P1, P2, P3):
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

    # Helper to get the expression var->coeff for a combo count c_i
    def combo_count_var_map(combo):
        info = combo_bits[combo]
        return {bit: w for bit, w in zip(info['bits'], info['weights'])}

    # 1) Team size: sum_i c_i == TEAM_SIZE
    # Build var_weight_map for all bits across combos
    team_var_weight = {}
    for combo in combos:
        team_var_weight.update(combo_count_var_map(combo))
    add_penalty_sum_equals(bqm, team_var_weight, TEAM_SIZE, P1)

    # 2) Weakness constraints: for each attack type j, sum_{i weak to j} c_i <= 2
    # We will convert to equality with slack S_j such that sum_weak + S_j = 2
    # Slack must be nonnegative integer. Upper bound for slack: if all combos (sum) can exceed 2 by at most TEAM_SIZE (worst case),
    # we can bound slack <= TEAM_SIZE. Encode slack with bits as well.
    attack_types = list(type_chart.columns)  # attacking types as columns
    # prepare a slack representation for this set (we can create separate slack bits per attack type)
    slack_bits_per_attack = {}
    slack_bits_needed = bits_needed(TEAM_SIZE)  # worst-case slack bound
    slack_weights = [2**i for i in range(slack_bits_needed)]
    for atk in attack_types:
        names = [f"S_weak_{atk}_b{j}" for j in range(slack_bits_needed)]
        slack_bits_per_attack[atk] = {'bits': names, 'weights': slack_weights}
    # For each attack type add penalty: (sum_{weak} c_i + slack - 2)^2 * P2
    for atk in attack_types:
        # find combos weak to this attacking type: multiplier > 1
        weak_combos = []
        for combo in combos:
            # combo is like 'Fire' or 'Fire/Flying'
            types = combo.split("/")
            # product of multipliers against each type for this attacking type
            # type_chart should be accessible: type_chart.loc[def_type, atk] or type_chart.at[def_type, atk]
            mult = 1.0
            for t in types:
                mult *= float(type_chart.at[t, atk])
            if mult > 1.0 + 1e-8:
                weak_combos.append(combo)
        # variable map: include counts for each weak combo, plus slack bits with positive weights
        var_map = {}
        for combo in weak_combos:
            var_map.update(combo_count_var_map(combo))
        # add slack bits
        slack_info = slack_bits_per_attack[atk]
        for bit, w in zip(slack_info['bits'], slack_info['weights']):
            var_map[bit] = w
        # RHS = 2
        add_penalty_sum_equals(bqm, var_map, 2, P2)

    # 3) Resistance constraints: for each attack type j, sum_{i resistant to j} c_i >= 1
    # Convert to equality by adding slack: sum_resist - (1 + S_j) = 0  => sum_resist - 1 - S_j = 0
    # Equivalent to sum_resist + T_j = 1 where T_j is nonnegative slack? We'll implement as sum_resist - (1 + S_j) = 0.
    slack_bits_per_attack_res = {}
    slack_bits_needed_res = bits_needed(TEAM_SIZE)  # upper bound
    slack_weights_res = [2**i for i in range(slack_bits_needed_res)]
    for atk in attack_types:
        names = [f"S_res_{atk}_b{j}" for j in range(slack_bits_needed_res)]
        slack_bits_per_attack_res[atk] = {'bits': names, 'weights': slack_weights_res}

    for atk in attack_types:
        resist_combos = []
        for combo in combos:
            types = combo.split("/")
            mult = 1.0
            for t in types:
                mult *= float(type_chart.at[t, atk])
            if mult < 1.0 - 1e-8:  # strictly resistant (mult < 1)
                resist_combos.append(combo)
        var_map = {}
        for combo in resist_combos:
            var_map.update(combo_count_var_map(combo))
        # add slack bits for S_j (we will encode S_j as nonnegative integer)
        slack_info = slack_bits_per_attack_res[atk]
        for bit, w in zip(slack_info['bits'], slack_info['weights']):
            var_map[bit] = -w  # because constraint is sum_resist - (1 + S) => coefficients of slack bits are -w
        # But the add_penalty_sum_equals expects var_weight_map a_i for sum Σ a_i x_i. To represent sum_resist - 1 - S = 0,
        # we will create a combined var map where S bits have coefficients (+w) but we'll set rhs = 1 and flip sign.
        # Simpler: create var_map_positive where slack bits are positive and we set rhs = 1 and add with penalty on (sum_resist + S - 1)^2.
        var_map2 = {}
        for combo in resist_combos:
            var_map2.update(combo_count_var_map(combo))
        for bit, w in zip(slack_info['bits'], slack_info['weights']):
            var_map2[bit] = w
        add_penalty_sum_equals(bqm, var_map2, 1, P3)

    return bqm

# Example run function tying everything together
def main():
    # load data
    pokemon_df = load_pokemon()
    type_chart = load_type_chart()
    combos = build_type_combos(pokemon_df)
    print("Type combos: ", combos)
    combo_bits = make_combo_bitnames(combos, MAX_PER_COMBO)
    bqm = build_bqm(combos, combo_bits, type_chart, P1, P2, P3)

    # solve (small problem) with simulated annealing
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=200)
    best = sampleset.first
    print("Best energy:", best.energy)
    sample = best.sample

    # reconstruct counts per combo
    def reconstruct_count(combo):
        info = combo_bits[combo]
        return sum(sample[bit] * w for bit, w in zip(info['bits'], info['weights']))

    team_counts = {combo: reconstruct_count(combo) for combo in combos}
    print("Team counts (combo: count):")
    for k, v in team_counts.items():
        if v > 0:
            print(f"  {k}: {v}")

    # you can expand counts to concrete Pokemon choices afterward by choosing actual Pokemon from each combo
    # (e.g., choose top stat/legendary constraints etc.)
    return

if __name__ == "__main__":
    main()
