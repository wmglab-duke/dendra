import re
from typing import Iterable

import torch


def matches_any_pattern(base_patterns: Iterable[str], target_string: str) -> bool:
    """
    Check whether a target string matches any dotted base pattern.

    A match occurs when each token in a pattern appears in order in the target
    string. All tokens except the final one must match entire words; the final
    token may match a word prefix.

    Additionally, the '*' character inside a pattern token is treated as a
    wildcard matching any sequence of characters (including empty).

    Parameters
    ----------
    base_patterns : Iterable[str]
        Collection of dot-separated pattern strings to test. Tokens may
        contain '*' as a wildcard.
    target_string : str
        Candidate string evaluated against each pattern.

    Returns
    -------
    bool
        True if any pattern matches the target string, False otherwise.

    Examples
    --------
    >>> matches_any_pattern(['hh.gbar'], 'hh.gbar_default')
    True
    >>> matches_any_pattern(['foo'], 'a.foo_bar')
    True
    >>> matches_any_pattern(['a.b'], 'a_b.c')
    False
    >>> matches_any_pattern(['*aug'], 'aug_default')
    True
    >>> matches_any_pattern(['*aug'], 'raug_default')
    True
    >>> matches_any_pattern(['*aug'], 'ina_aug_default')
    True
    """

    def _pattern_part_to_regex(part: str) -> str:
        # Escape everything, then turn escaped '*' (r'\*') back into '.*'
        escaped = re.escape(part)
        return escaped.replace(r"\*", ".*")

    for base_pattern in base_patterns:
        # Split the pattern by '.' and convert each part, treating '*' as wildcard.
        regex_parts = [_pattern_part_to_regex(part) for part in base_pattern.split(".")]

        # The separator `\b.*?\b` ensures that all intermediate parts are
        # treated as whole words.
        regex_pattern = (
            r"\b"  # The pattern must start at a word boundary.
            + r"\b.*?\b".join(regex_parts)
            # No trailing \b so the final token may match a word prefix.
        )

        if re.search(regex_pattern, target_string, re.IGNORECASE):
            return True

    return False


class AxModule(torch.nn.Module):
    """Base class for modular neuron model components."""

    def unfreeze(self, *names):
        """
        Unfreezes model parameters, making them trainable.

        If no names are provided, all parameters will be unfrozen.
        If names are provided, only parameters whose names match any
        of the provided patterns will be unfrozen.

        Parameters
        ----------
        *names : str
            Variable length argument list of parameter name patterns.
            If empty, all parameters will be unfrozen.
            Otherwise, only parameters matching any of these patterns will be unfrozen.

        Notes
        -----
        The matching is done using the `matches_any_pattern` function.
        When a parameter is unfrozen, a message is printed to the console.
        """
        if not names:
            for p in self.parameters():
                p.requires_grad = True
        else:
            for n, p in self.named_parameters():
                if matches_any_pattern(names, n):
                    print(f"Unfreezing {n}")
                    p.requires_grad = True
        return self

    def unfreeze_(self, *names):
        """
        In-place alias of :meth:`unfreeze`.

        Parameters
        ----------
        *names : str
            Optional name patterns forwarded to :meth:`unfreeze`.
        """
        self.unfreeze(*names)

    def freeze(self, *names):
        """
        Freeze parameters to disable gradient computation.

        Parameters
        ----------
        *names : str
            Optional name patterns selecting parameters to freeze. When omitted,
            all parameters are frozen.

        Returns
        -------
        Population
            The population instance for chaining.
        """
        if not names:
            for p in self.parameters():
                p.requires_grad = False
        else:
            for n, p in self.named_parameters():
                if matches_any_pattern(names, n):
                    print(f"Freezing {n}")
                    p.requires_grad = False
        return self

    def freeze_(self, *names):
        """
        In-place alias of :meth:`freeze`.

        Parameters
        ----------
        *names : str
            Optional name patterns forwarded to :meth:`freeze`.
        """
        self.freeze(*names)
