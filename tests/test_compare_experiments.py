import sys

import pytest

sys.path.insert(0, "src")

import compare_experiments


def test_resolve_experiment_name_prefers_explicit_name():
    result = compare_experiments.resolve_experiment_name(
        "Manual Experiment",
        "initial",
        "initial",
    )

    assert result == "Manual Experiment"


def test_resolve_experiment_name_uses_suite_and_scenario(monkeypatch):
    monkeypatch.setattr(
        compare_experiments,
        "get_scenario_title",
        lambda suite_name, scenario_name: f"{suite_name}:{scenario_name}",
    )

    result = compare_experiments.resolve_experiment_name(
        None,
        "initial",
        "ci",
    )

    assert result == "initial:ci"


def test_resolve_experiment_name_raises_without_required_inputs():
    with pytest.raises(ValueError, match="Provide --experiment-name"):
        compare_experiments.resolve_experiment_name(None, "initial", None)
