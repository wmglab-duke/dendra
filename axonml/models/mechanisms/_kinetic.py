# kinetic_to_derivatives_nmodl.py
import re
from typing import Dict, List, NamedTuple, Optional, Tuple

# ---------- parsing utilities ----------

_SPEC_RE = re.compile(r"^\s*(?:(\d+)\s*)?([A-Za-z_]\w*)\s*$")


def _parse_side(side: str, known: set) -> Dict[str, int]:
    """Parse 'A + 3B + 2C' -> {'A':1, 'B':3, 'C':2}. Empty -> {}."""
    side = side.strip()
    if not side:
        return {}
    coeffs: Dict[str, int] = {}
    for chunk in side.split("+"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _SPEC_RE.match(chunk)
        if not m:
            raise ValueError(f"Bad species term '{chunk}'")
        n_str, name = m.groups()
        if name not in known:
            raise ValueError(f"Unknown state '{name}' (not in provided states)")
        n = int(n_str) if n_str else 1
        coeffs[name] = coeffs.get(name, 0) + n
    return coeffs


def _last_paren_block(s: str) -> Tuple[str, str]:
    """Split '... (rates)' -> ('...', 'rates') using the last (...) block."""
    end = s.rfind(")")
    if end < 0:
        raise ValueError("Missing closing ')'")
    depth, start = 0, None
    for i in range(end, -1, -1):
        if s[i] == ")":
            depth += 1
        elif s[i] == "(":
            depth -= 1
            if depth == 0:
                start = i
                break
    if start is None:
        raise ValueError("Unmatched parentheses in rate block")
    head = s[:start].rstrip()
    inside = s[start + 1 : end].strip()
    tail = s[end + 1 :].strip()
    if tail and not tail.isspace():
        raise ValueError(f"Trailing garbage after rate block: {tail!r}")
    return head, inside


def _split_top_level_commas(s: str) -> List[str]:
    """Split on commas that are NOT inside parentheses."""
    out, cur, depth = [], [], 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return out


def _mass_action_factor(stoich: Dict[str, int]) -> str:
    """Return product like 'A * A * B' or '1' if empty."""
    terms: List[str] = []
    for sp, n in stoich.items():
        terms.extend([sp] * n)
    return " * ".join(terms) if terms else "1"


# ---------- CONSERVE handling ----------


class ConserveSpec(NamedTuple):
    weights: Dict[str, int]  # e.g., {'C':1, 'O':1, 'I':1}
    const: str  # e.g., '1' or 'Total'


def _parse_conserve(line: str, states: List[str]) -> ConserveSpec:
    """Parse 'CONSERVE C + O + I = 1' (or just 'C + O + I = 1')."""
    s = line.strip()
    if s.upper().startswith("CONSERVE"):
        s = s[len("CONSERVE") :].strip()
    if "=" not in s:
        raise ValueError("CONSERVE line must contain '='")
    lhs, rhs = s.split("=", 1)
    weights = _parse_side(lhs, set(states))
    const = rhs.strip()
    if not const:
        raise ValueError("Right-hand side of CONSERVE is empty")
    if not weights:
        raise ValueError("Left-hand side of CONSERVE is empty")
    return ConserveSpec(weights, const)


def _apply_conserve_to_derivs(
    states: List[str],
    deriv_map: Dict[str, str],  # X -> RHS(X) string (no outer parens)
    cons: ConserveSpec,
    eliminate_var: Optional[str] = None,
) -> Tuple[Dict[str, str], str]:
    """
    Enforce sum_i w_i x_i' = 0 by setting one x_j' = -(1/w_j) * sum_{i≠j} w_i x_i'.
    Returns (new_deriv_map, eliminated_var_name).
    """
    involved = [s for s in states if s in cons.weights and cons.weights[s] != 0]
    if not involved:
        raise ValueError("CONSERVE has no overlap with states")
    elim = eliminate_var or involved[-1]  # default: last by 'states' order
    if elim not in cons.weights or cons.weights[elim] == 0:
        raise ValueError(
            f"Chosen eliminate='{elim}' not in CONSERVE variables (or weight=0)"
        )

    wj = cons.weights[elim]
    parts = []
    for X in involved:
        if X == elim:
            continue
        wi = cons.weights[X]
        rhsX = deriv_map.get(X, "0")
        parts.append(f"{wi}*({rhsX})")
    elim_rhs = f"-({' + '.join(parts)})/{wj}" if parts else "0"

    new_map = dict(deriv_map)
    new_map[elim] = elim_rhs
    return new_map, elim


def make_conserve_enforcer(cons: ConserveSpec, eliminate_var: Optional[str] = None):
    """
    Build a runtime enforcer for sum_i w_i x_i = const:
      x_elim = (const - sum_{i≠elim} w_i x_i) / w_elim
    Returns (elim_var, algebraic_assignment_str, enforcer(mapping, env)).
    """
    weights = dict(cons.weights)
    names = [k for k, w in weights.items() if w != 0]
    if not names:
        raise ValueError("All weights are zero in CONSERVE")

    elim_var = eliminate_var or names[-1]
    if elim_var not in weights or weights[elim_var] == 0:
        raise ValueError(f"Eliminated variable '{elim_var}' missing or has zero weight")

    others = [n for n in names if n != elim_var]
    if others:
        sum_others = " + ".join(
            [f"{weights[n]}*{n}" if weights[n] != 1 else f"{n}" for n in others]
        )
        num = f"({cons.const} - ({sum_others}))"
    else:
        num = f"({cons.const})"
    denom = f"{weights[elim_var]}"
    assign_str = f"{elim_var} = {num} / ({denom})"

    def enforcer(mapping: Dict[str, object], env: Optional[Dict[str, object]] = None):
        scope = {}
        scope.update(mapping)
        if env:
            scope.update(env)
        const_val = eval(cons.const, {}, scope)  # trusted strings only
        total = None
        for n in others:
            wi = weights[n]
            term = mapping[n]
            total = wi * term if total is None else total + wi * term
        numer = const_val if total is None else const_val - total
        mapping[elim_var] = numer / weights[elim_var]

    return elim_var, assign_str, enforcer


# ---------- main: reactions + CONSERVE-in-kinetic ----------


def kinetic_to_derivatives(
    states: List[str],
    kinetic: List[str],
    eliminate: Optional[Dict[int, str]] = None,  # map CONSERVE-index -> var name
):
    """
    Parse NMODL-style KINETIC lines (reactions + CONSERVE) into derivative statements.

    Args
    ----
    states    : list of species names.
    kinetic   : list of lines; each is either a reaction starting with '~'
                or a CONSERVE line ('CONSERVE ...' or 'conserve ...').
    eliminate : optional dict telling which variable to eliminate for each CONSERVE,
                keyed by the order that CONSERVE lines appear in `kinetic` (0-based).

    Returns
    -------
    deriv_list    : ["X' = (...)", ...] in the order of `states`.
    deriv_map     : {X: "(...)"}
    conserve_info : list of (elim_var, algebraic_assignment_str) in the same order
                    as CONSERVE lines encountered.
    """
    known = set(states)
    rhs_terms: Dict[str, List[str]] = {s: [] for s in states}
    conserve_specs: List[ConserveSpec] = []
    conserve_order: List[int] = []  # indices in original kinetic list (optional info)

    for idx, raw in enumerate(kinetic):
        s = raw.strip()
        if not s:
            continue

        # CONSERVE line?
        if s.upper().startswith("CONSERVE"):
            cons = _parse_conserve(s, states)
            conserve_specs.append(cons)
            conserve_order.append(idx)
            continue

        # Reaction line must start with '~'
        if not s.startswith("~"):
            raise ValueError(f"Expected '~' reaction or 'CONSERVE' line: {raw!r}")
        s = s[1:].strip()

        # Choose arrow
        if "<->" in s:
            arrow = "<->"
        elif "->" in s:
            arrow = "->"
        else:
            raise ValueError(f"Missing '->' or '<->' in: {raw!r}")

        lhs, right_part = s.split(arrow, 1)
        rhs_side, rates_inside = _last_paren_block(right_part)
        rates = _split_top_level_commas(rates_inside)
        if arrow == "<->":
            if len(rates) != 2:
                raise ValueError(
                    f"Reversible reaction needs two rates '(kf, kb)': {raw!r}"
                )
            kf, kb = rates[0], rates[1]
        else:
            if len(rates) != 1:
                raise ValueError(
                    f"Unidirectional reaction needs one rate '(k)': {raw!r}"
                )
            kf, kb = rates[0], None

        nu_L = _parse_side(lhs, known)
        nu_R = _parse_side(rhs_side, known)

        fac_f = _mass_action_factor(nu_L)
        v_f = f"({kf})" if fac_f == "1" else f"({kf}*{fac_f})"
        v_b = None
        if kb is not None:
            fac_b = _mass_action_factor(nu_R)
            v_b = f"({kb})" if fac_b == "1" else f"({kb}*{fac_b})"

        for X in states:
            d = nu_R.get(X, 0) - nu_L.get(X, 0)
            if d == 0:
                continue
            if v_b is None:
                rhs_terms[X].append(f"{d}*{v_f}")
            else:
                rhs_terms[X].append(f"{d}*({v_f} - {v_b})")

    deriv_map: Dict[str, str] = {
        X: (" + ".join(rhs_terms[X]) if rhs_terms[X] else "0") for X in states
    }

    conserve_info: List[Tuple[str, str]] = []
    for j, cons in enumerate(conserve_specs):
        elim_var = (eliminate or {}).get(j, None)
        deriv_map, eliminated = _apply_conserve_to_derivs(
            states, deriv_map, cons, eliminate_var=elim_var
        )
        elim2, assign_str, _ = make_conserve_enforcer(
            cons, eliminate_var=elim_var or eliminated
        )
        conserve_info.append((elim2, assign_str))

    deriv_list = [f"{X}' = ({deriv_map[X]})" for X in states]

    if not conserve_info:
        conserve_info = None

    return deriv_list, deriv_map, conserve_info
