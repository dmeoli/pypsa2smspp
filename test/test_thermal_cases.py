# -*- coding: utf-8 -*-
"""
The unit commitment cases of PyPSA that a ThermalUnitBlock writes.

PyPSA gives a `Generator` more than the model of the D3.2 has: a cost for
shutting down, an output before the horizon that may be unknown, a unit that
is never committed, a bound on the energy of the whole horizon and modules of
the capacity. Here the translation of each of them is read back from the
netCDF file, which needs no SMS++, while the tests that compare the two
objectives need one and live in `test_ucblock.py` and `test_nuclear.py`.
"""
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pypsa
import pytest

from conftest import safe_remove, OUT_TEST

from pypsa2smspp.transformation import Transformation
from pypsa2smspp.utils import fix_commitment_on, free_initial_ramp


SNAPSHOTS = 12


def thermal_network(committable: bool = True, **generator) -> pypsa.Network:
    """One bus, one thermal generator with the given attributes, and a load."""
    n = pypsa.Network()
    n.set_snapshots(range(SNAPSHOTS))
    n.add("Bus", "bus")
    n.add("Carrier", ["ocgt", "slack"])
    n.add("Load", "load", bus="bus",
          p_set=600.0 + 200.0 * np.cos(2.0 * np.pi * np.arange(SNAPSHOTS) / 6.0))
    n.add("Generator", "unit", bus="bus", carrier="ocgt", p_nom=1000.0,
          marginal_cost=40.0, p_min_pu=0.3, ramp_limit_up=0.2,
          ramp_limit_down=0.2, committable=committable,
          **({"up_time_before": 1} if committable else {}), **generator)
    n.add("Generator", "shedding", bus="bus", carrier="slack", p_nom=1e5,
          marginal_cost=3000.0)
    return n


def unit_variables(network: pypsa.Network, case_name: str) -> dict:
    """The variables of the ThermalUnitBlock the network is translated into."""
    temp_nc = OUT_TEST / f"thermal_{case_name}.nc"
    safe_remove(temp_nc)

    transformation = Transformation(capacity_expansion_ucblock=True,
                                    enable_thermal_units=True,
                                    intermittent_carriers=[])
    transformation.create_model(network, verbose=False)
    transformation.sms_network.to_netcdf(temp_nc, force=True)

    with nc.Dataset(temp_nc) as dataset:
        for group in dataset["Block_0"].groups.values():
            if getattr(group, "type", None) != "ThermalUnitBlock":
                continue
            if float(np.ravel(group["MaxPower"][...])[0]) != 1000.0:
                continue   # the slack unit, not the one under test
            return {name: np.ravel(variable[...])
                    for name, variable in group.variables.items()}

    raise AssertionError("no ThermalUnitBlock of the generator in the file")


def test_shut_down_cost():
    variables = unit_variables(thermal_network(shut_down_cost=700.0),
                               "shut_down_cost")

    # PyPSA charges a shut-down once, whatever the weighting of the snapshot
    assert np.all(variables["ShutDownCost"] == 700.0)


def test_no_shut_down_cost():
    variables = unit_variables(thermal_network(), "no_shut_down_cost")

    assert "ShutDownCost" not in variables


def test_free_initial_ramp():
    # the network gives no p_init, so nothing of the first instant binds: the
    # initial power is the maximum power there and the ramps are as wide as
    # the whole range
    variables = unit_variables(thermal_network(), "free_initial_ramp")

    assert variables["InitialPower"][0] == 1000.0
    for name in ("DeltaRampUp", "DeltaRampDown"):
        assert variables[name][0] == 1000.0
        assert variables[name][1] == 200.0
    # the shut-down limit of the first instant does not bind either, the unit
    # having none of its own (PyPSA then allows the whole nominal power)
    assert variables["ShutDownLimit"][0] == 1000.0


def test_initial_ramp_with_p_init():
    variables = unit_variables(thermal_network(p_init=450.0), "p_init")

    assert variables["InitialPower"][0] == 450.0
    assert np.all(variables["DeltaRampUp"] == 200.0)


def test_commitment_fixed_when_not_committable():
    variables = unit_variables(thermal_network(committable=False),
                               "not_committable")

    # on before the horizon and a minimum up time longer than it: the unit
    # never switches, as a generator PyPSA does not commit
    assert variables["InitUpDownTime"][0] == 1
    assert variables["MinUpTime"][0] == SNAPSHOTS + 1
    assert "StartUpCost" not in variables


def test_energy_bound_refused():
    with pytest.raises(ValueError, match="e_sum_max"):
        unit_variables(thermal_network(e_sum_max=1e4), "e_sum")


def test_modular_unit_refused():
    with pytest.raises(ValueError, match="p_nom_mod"):
        unit_variables(thermal_network(p_nom_mod=250.0), "p_nom_mod")


def test_helpers_on_their_own():
    variables = {"MaxPower": {"value": np.array([300.0, 500.0])},
                 "DeltaRampUp": {"value": 100.0, "size": 1},
                 "InitialPower": {"value": 500.0}}
    free_initial_ramp(variables, 2)
    assert list(variables["DeltaRampUp"]["value"]) == [500.0, 100.0]
    assert variables["InitialPower"]["value"] == 300.0

    fix_commitment_on(variables, 2)
    assert variables["MinUpTime"]["value"] == 3
    assert variables["InitUpDownTime"]["value"] == 1
