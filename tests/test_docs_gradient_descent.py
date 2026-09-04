"""Exercise the imperative optimization tutorial without its full training cost."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import dendra as dn
from dendra.models.core import Population

NOTEBOOK = Path(__file__).parents[1] / "docs" / "basics" / "04_gradient_descent.ipynb"


def test_gradient_descent_notebook_executes_from_fresh_namespace(monkeypatch):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    original_run = Population.run
    trainable_at_last_run = {}
    figures = []

    def short_run(model, *args, **kwargs):
        # Include the complete 0.1--0.3 ms stimulus, but omit the long tail.
        kwargs["tstop"] = min(kwargs["tstop"], 0.4)
        trainable_at_last_run[id(model)] = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        return original_run(model, *args, **kwargs)

    def short_range(*args):
        return range(2) if args == (500,) else range(*args)

    def headless_subplots(*args, **kwargs):
        # Avoid changing pyplot's backend or closing another test's figures.
        figure = Figure(figsize=kwargs.pop("figsize", None))
        FigureCanvasAgg(figure)
        figures.append(figure)
        return figure, figure.subplots(*args, **kwargs)

    monkeypatch.setattr(Population, "run", short_run)
    monkeypatch.setattr(dn, "set_jit_enabled", lambda _enabled: None)
    monkeypatch.setattr(plt, "subplots", headless_subplots)
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {"__name__": "__main__", "range": short_range}
    initial_parameters = None

    try:
        with dn.ctx(JIT=0):
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                exec(compile(source, f"{NOTEBOOK}:cell-{index}", "exec"), namespace)

                if initial_parameters is None and "loss" in namespace:
                    candidate = namespace["bad_model"]
                    initial_parameters = {}
                    assert torch.isfinite(namespace["loss"])
                    for name, parameter in candidate.named_parameters():
                        if parameter.requires_grad:
                            initial_parameters[name] = parameter.detach().clone()
                            assert parameter.grad is not None
                            assert torch.isfinite(parameter.grad).all()
                            assert torch.count_nonzero(parameter.grad) > 0

            candidate = namespace["bad_model"]
            final_parameters = {
                name: parameter.detach()
                for name, parameter in candidate.named_parameters()
                if parameter.requires_grad
            }
            assert initial_parameters
            assert final_parameters.keys() == initial_parameters.keys()
            for name, parameter in final_parameters.items():
                assert torch.isfinite(parameter).all()
                assert not torch.equal(parameter, initial_parameters[name])
                # The final plot must use a simulation after the last update,
                # not the trace that produced that update's loss.
                torch.testing.assert_close(
                    trainable_at_last_run[id(candidate)][name],
                    parameter,
                    rtol=0,
                    atol=0,
                )
            assert torch.isfinite(namespace["v_bad"]).all()
            torch.testing.assert_close(namespace["v_bad"][-1], candidate.v)
            assert len(figures) == 2
    finally:
        for figure in figures:
            figure.clear()
