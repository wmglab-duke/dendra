# kinetic_to_derivatives_nmodl.py
import re
from typing import Dict, List, NamedTuple, Optional, Tuple

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


# ------------------------- CONSERVE parsing/apply -------------------------


class ConserveSpec(NamedTuple):
    weights: Dict[str, int]  # e.g., {'C':1, 'O':1, 'I':1}
    const: str  # e.g., '1' or 'Tot'


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
    amount_map: Dict[str, str],  # X -> (amount/time) RHS (pre-divide by compartment)
    comp: Dict[str, str],  # X -> compartment expr (or "1")
    cons: ConserveSpec,
    eliminate_var: Optional[str] = None,
) -> Tuple[Dict[str, str], str]:
    """
    Enforce   sum_i w_i * (amount_i) = 0,   where amount_i = comp_i * x_i'
    by solving for the eliminated state's amount term.

    amount_j = -(1/w_j) * sum_{i≠j} w_i * amount_i
    """
    involved = [s for s in states if cons.weights.get(s, 0) != 0]
    if not involved:
        raise ValueError("CONSERVE has no overlap with states")
    elim = eliminate_var or involved[-1]
    if cons.weights.get(elim, 0) == 0:
        raise ValueError(
            f"Chosen eliminate='{elim}' not in CONSERVE vars (or weight=0)"
        )

    wj = cons.weights[elim]
    parts = []
    for X in involved:
        if X == elim:
            continue
        wi = cons.weights[X]
        parts.append(f"{wi}*({amount_map.get(X, '0')})")
    rhs_amount_elim = f"-({' + '.join(parts)})/({wj})" if parts else "0"

    new_map = dict(amount_map)
    new_map[elim] = rhs_amount_elim
    return new_map, elim


def make_conserve_enforcer(
    cons: ConserveSpec, comp: Dict[str, str], eliminate_var: Optional[str] = None
):
    """
    Build a runtime enforcer for:  sum_i w_i * comp_i * x_i = const
      x_elim = (const - sum_{i≠elim} w_i*comp_i*x_i) / (w_elim*comp_elim)
    Returns (elim_var, algebraic_assignment_str)
    """
    names = [k for k, w in cons.weights.items() if w != 0]
    if not names:
        raise ValueError("All weights are zero in CONSERVE")
    elim_var = eliminate_var or names[-1]
    if cons.weights.get(elim_var, 0) == 0:
        raise ValueError(f"Eliminated variable '{elim_var}' missing or has zero weight")

    wj = cons.weights[elim_var]
    cj = comp.get(elim_var, "1")
    others = [n for n in names if n != elim_var]
    if others:
        sum_others = " + ".join(
            [f"{cons.weights[n]}*({comp.get(n, '1')})*{n}" for n in others]
        )
        num = f"({cons.const} - ({sum_others}))"
    else:
        num = f"({cons.const})"
    denom = f"{wj}*({cj})"
    assign_str = f"{elim_var} = {num} / ({denom})"
    return elim_var, assign_str


# --------------------------- COMPARTMENT parsing ---------------------------


def _parse_compartment(line: str, states: List[str]) -> Dict[str, str]:
    """
    Parse 'COMPARTMENT vol {a b c}' or 'COMPARTMENT i, vol[i] {a}'.
    Returns mapping {state: vol_expr}. Later statements override earlier ones.
    """
    s = line.strip()
    if not s.upper().startswith("COMPARTMENT"):
        raise ValueError("Not a COMPARTMENT line")
    s = s[len("COMPARTMENT") :].strip()
    lb = s.find("{")
    rb = s.rfind("}")
    if lb < 0 or rb < 0 or rb < lb:
        raise ValueError("COMPARTMENT must specify states inside braces {...}")
    states_str = s[lb + 1 : rb].strip()
    vol_part = s[:lb].strip()
    if "," in vol_part:
        vol_expr = vol_part.split(",", 1)[1].strip()
    else:
        vol_expr = vol_part
    if not vol_expr:
        raise ValueError("Empty volume expression in COMPARTMENT")
    names = [t for t in states_str.replace(",", " ").split() if t]
    known = set(states)
    out = {}
    for n in names:
        if n not in known:
            raise ValueError(f"Unknown state '{n}' in COMPARTMENT")
        out[n] = vol_expr
    return out


# ----------------------- main: reactions/flux + CONSERVE + COMP -----------


def kinetic_to_derivatives(
    states: List[str],
    kinetic: List[str],
    eliminate: Optional[Dict[int, str]] = None,  # map CONSERVE-index -> eliminated var
):
    """
    Parse NMODL-style KINETIC lines (reactions '~', explicit flux '<<',
    COMPARTMENT, CONSERVE) into:
      - concentration ODEs,
      - amount (pre-divide) RHS,
      - compartment map,
      - CONSERVE algebraic assignments,
      - an ordered 'flux_log' describing each reaction/flux.

    Returns
    -------
    deriv_list    : ["X' = (...)", ...]  (concentration ODEs)
    deriv_map     : {X: "(amount/time RHS BEFORE dividing by compartment)"}
    conserve_info : [(elim_var, "elim_var = ...")]
    comp_map      : {X: "compartment_expr_or_1"}
    flux_log      : [ { ... }, ... ]   (see below)
    """
    known = set(states)
    rhs_amount_terms: Dict[str, List[str]] = {s: [] for s in states}
    comp_map: Dict[str, str] = {s: "1" for s in states}
    conserve_specs: List[ConserveSpec] = []
    flux_log: List[Dict[str, str]] = []

    for raw in kinetic:
        line = raw.strip()
        if not line:
            continue

        U = line.upper()
        if U.startswith("CONSERVE"):
            conserve_specs.append(_parse_conserve(line, states))
            continue
        if U.startswith("COMPARTMENT"):
            comp_map.update(_parse_compartment(line, states))
            continue
        if U.startswith("LONGITUDINAL_DIFFUSION"):
            # Axial coupling (ignored here; contributes to global sparse solve, not local ODE).
            continue

        # Reaction/flux lines must start with '~'
        if not line.startswith("~"):
            raise ValueError(
                f"Expected '~' reaction/flux, 'COMPARTMENT', or 'CONSERVE': {raw!r}"
            )
        s = line[1:].strip()

        # Explicit flux: "~ X << expr"  (exactly one target, with optional integer coeff)
        if "<<" in s and ("->" not in s and "<->" not in s):
            lhs, flux = s.split("<<", 1)
            lhs = lhs.strip()
            flux = flux.strip()
            sto = _parse_side(lhs, known)
            if not sto or len(sto) != 1:
                raise ValueError(
                    f"Flux '<<' must target exactly one state; got '{lhs}'"
                )
            ((X, coeff),) = sto.items()
            term = f"{coeff}*({flux})"
            rhs_amount_terms[X].append(term)
            flux_log.append(
                {
                    "kind": "explicit_flux",
                    "raw": raw,
                    "target": X,
                    "coeff": str(coeff),
                    "flux": flux,
                    "amount_term": term,
                }
            )
            continue

        # Mass-action reaction: "<->" or "->"
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
        if kb is not None:
            fac_b = _mass_action_factor(nu_R)
            v_b = f"({kb})" if fac_b == "1" else f"({kb}*{fac_b})"
        else:
            v_b = "0"

        # Bookkeeping log for this reaction
        flux_log.append(
            {
                "kind": "reaction",
                "raw": raw,
                "lhs": nu_L,
                "rhs": nu_R,
                "kf": kf,
                "kb": (kb if kb is not None else "0"),
                "f_flux": v_f,
                "b_flux": v_b,
            }
        )

        # Amount-balance contributions: dX/dt (amount) += (nu_R[X]-nu_L[X]) * (f_flux - b_flux)
        contrib = f"(({v_f}) - ({v_b}))"
        for X in states:
            d = nu_R.get(X, 0) - nu_L.get(X, 0)
            if d != 0:
                rhs_amount_terms[X].append(f"{d}*{contrib}")

    # Unscaled amount map
    amount_map: Dict[str, str] = {
        X: (" + ".join(rhs_amount_terms[X]) if rhs_amount_terms[X] else "0")
        for X in states
    }

    # Apply CONSERVE constraints in amount form
    conserve_info: List[Tuple[str, str]] = []
    for j, cons in enumerate(conserve_specs):
        elim_var = (eliminate or {}).get(j, None)
        amount_map, eliminated = _apply_conserve_to_derivs(
            states, amount_map, comp_map, cons, eliminate_var=elim_var
        )
        elim2, assign_str = make_conserve_enforcer(
            cons, comp_map, eliminate_var=elim_var or eliminated
        )
        conserve_info.append((elim2, assign_str))

    # Divide by compartments to get concentration ODEs
    deriv_map: Dict[str, str] = {
        X: f"({amount_map[X]})/({comp_map.get(X, '1')})" for X in states
    }
    deriv_list = [f"{X}' = ({deriv_map[X]})" for X in states]

    if not conserve_info:
        conserve_info = None

    return deriv_list, amount_map, conserve_info, comp_map, flux_log
