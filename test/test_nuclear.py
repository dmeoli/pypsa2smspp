# -*- coding: utf-8 -*-
"""
NuclearUnitBlock regression test.

The nuclear generators of a small single-bus network are translated into
NuclearUnitBlocks through the `nuclear_units` option:

- with rules that do not bind (the full thermal ramps allowed outside the
  modulations and no other rule) the unit is a ThermalUnitBlock, so the
  SMS++ objective must match the PyPSA one;
- with the default rules PyPSA solves a relaxation of the SMS++ problem, so
  the SMS++ objective must not be below the PyPSA one.

Both cases also check that the operating rules are written in the netCDF file.
"""
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pypsa
import pytest

from conftest import safe_remove, REL_TOL, ABS_TOL, OUT_TEST

from pypsa2smspp.constants import nuclear_rules_default
from pypsa2smspp.transformation import Transformation
from pypsa2smspp.utils import forbid_unreachable_switches


# the rules of nuclear_rules_default switched off, the modulation ramps being
# the full thermal ones: a NuclearUnitBlock that behaves as a ThermalUnitBlock
NON_BINDING_RULES = {key: None for key in nuclear_rules_default}
NON_BINDING_RULES.update(modulation_ramp_fraction=1.0, modulation_time=2.0)


def nuclear_network(snapshots: int = 48) -> pypsa.Network:
    """
    One bus, four committable nuclear units that start on, a peaking unit that
    starts off, and a load shedding generator; the load follows a daily cycle
    whose peak is at the first snapshot, so that the initial power of the
    nuclear units is reachable.
    """
    n = pypsa.Network()
    n.set_snapshots(range(snapshots))
    n.add("Bus", "bus")
    n.add("Carrier", ["nuclear", "ocgt", "slack"])

    t = np.arange(snapshots)
    n.add("Load", "load", bus="bus",
          p_set=3000.0 + 1200.0 * np.cos(2.0 * np.pi * t / 24.0))

    for i, p_nom in enumerate([1300.0, 1300.0, 900.0, 900.0]):
        n.add("Generator", f"nuclear{i}", bus="bus", carrier="nuclear",
              p_nom=p_nom, marginal_cost=9.0 + i, committable=True,
              p_min_pu=0.5, ramp_limit_up=0.3, ramp_limit_down=0.3,
              ramp_limit_start_up=0.5, ramp_limit_shut_down=0.5,
              min_up_time=4, min_down_time=4, start_up_cost=250.0 * p_nom,
              up_time_before=10)

    n.add("Generator", "ocgt", bus="bus", carrier="ocgt", p_nom=2000.0,
          marginal_cost=80.0, committable=True, p_min_pu=0.2,
          start_up_cost=20000.0, up_time_before=0, down_time_before=10)
    n.add("Generator", "shedding", bus="bus", carrier="slack", p_nom=1e5,
          marginal_cost=3000.0)
    return n


def run_nuclear(case_name: str, rules) -> tuple[float, float, Path]:
    """
    Solves the network with PyPSA and with SMS++, the nuclear units being
    NuclearUnitBlocks with the given rules; returns both objectives and the
    SMS++ input file.
    """
    temp_nc = OUT_TEST / f"smspp_{case_name}_temp.nc"
    safe_remove(temp_nc)

    network = nuclear_network()
    network.optimize(solver_name="highs")
    obj_pypsa = float(network.objective)

    transformation = Transformation(
        capacity_expansion_ucblock=True,
        enable_thermal_units=True,
        intermittent_carriers=[],
        nuclear_units={"nuclear": rules},
        workdir=OUT_TEST,
        name=case_name,
        overwrite=True,
        fp_temp="smspp_{name}_temp.nc",
        fp_log="smspp_{name}_log.txt",
        fp_solution="smspp_{name}_solution.nc",
        configfile="auto",
    )
    transformation.run(nuclear_network(), verbose=False)

    return obj_pypsa, float(transformation.result.objective_value), temp_nc


def nuclear_groups(temp_nc: Path) -> list:
    """The unit groups of the SMS++ file whose type is NuclearUnitBlock."""
    groups = []

    def visit(group):
        if getattr(group, "type", None) == "NuclearUnitBlock":
            groups.append(group)
        for child in group.groups.values():
            visit(child)

    with nc.Dataset(temp_nc) as dataset:
        visit(dataset)
        return [(g.name, set(g.variables)) for g in groups]


def test_nuclear_non_binding_rules():
    obj_pypsa, obj_smspp, temp_nc = run_nuclear("nuclear_non_binding",
                                                NON_BINDING_RULES)

    groups = nuclear_groups(temp_nc)
    assert len(groups) == 4
    for _, variables in groups:
        assert {"ModulationTime", "ModulationDeltaRampUp",
                "ModulationDeltaRampDown"} <= variables
        assert "PowerBands" not in variables

    assert obj_smspp == pytest.approx(obj_pypsa, rel=REL_TOL, abs=ABS_TOL)


def test_nuclear_default_rules():
    obj_pypsa, obj_smspp, temp_nc = run_nuclear("nuclear_default", True)

    groups = nuclear_groups(temp_nc)
    assert len(groups) == 4
    for _, variables in groups:
        assert {"ModulationTime", "MaxModulationLength", "DayLength",
                "ModulationsPerDay", "StartUpsPerDay", "PowerBands",
                "DeepDecreaseThreshold", "DeepDecreaseGradient"} <= variables

    assert obj_smspp >= obj_pypsa * (1.0 - REL_TOL) - ABS_TOL


def test_nuclear_unknown_rule():
    with pytest.raises(ValueError, match="Unknown nuclear rules"):
        Transformation(enable_thermal_units=True,
                       nuclear_units={"nuclear": {"modulation_per_day": 2}})


def test_unreachable_switches():
    # a unit down for 3 instants whose start-up limit is below its minimum
    # power cannot start up in a horizon of 24 instants; its shut-down limit is
    # above the minimum power and nothing changes on that side
    variables = {"MinPower": {"value": np.array([500.0])},
                 "StartUpLimit": {"value": 200.0},
                 "ShutDownLimit": {"value": 600.0},
                 "InitUpDownTime": {"value": -3},
                 "MinDownTime": {"value": 1},
                 "MinUpTime": {"value": 1}}
    forbid_unreachable_switches(variables, 24)

    assert variables["StartUpLimit"]["value"] == 500.0
    assert variables["MinDownTime"]["value"] == 27
    assert variables["ShutDownLimit"]["value"] == 600.0
    assert variables["MinUpTime"]["value"] == 1


if __name__ == "__main__":
    test_nuclear_non_binding_rules()
    test_nuclear_default_rules()
