# -*- coding: utf-8 -*-
"""
Generator of the resilient UCBlock instances with pollutant budget constraints.

Each instance is one of the Excel test networks whose carriers are given an
attribute (e.g., an emission rate) and whose dispatch is bounded by PyPSA
GlobalConstraint on it, set to a fraction of the value of the unconstrained
dispatch: primary energy limits from above, from below and with equality, an
operational limit, and primary energy limits where a store and a storage unit
(both not cyclic) contribute through their state of charge. The network is
solved with PyPSA, which
gives the reference objective value, and converted by pypsa2smspp, where each
limit becomes a pollutant budget constraint of the UCBlock; the netCDF file of
the UCBlock is written in the output directory, and the reference values are
printed in the format of the REF_OBJ entries of the SMS++ batch files.

The extendable assets whose bound is infinite are given one that comes from
the demand of the network itself (bound_extendable_assets of pypsa2smspp),
since a design with no bound makes the Lagrangian subproblem unbounded while a
bound picked out of thin air is worse than no bound at all: the design is
bang-bang, the value of the component becomes the bound times the investment
cost, and the master problem of the bundle ends up with coefficients its
quadratic term cannot be compared with. SMSPP_DESIGN_BOUNDS chooses among the
three ways of writing the instances, "physical" (the default), "none" and
"sentinel", so that the same network can be run in all of them.

Note that the Excel networks are not deterministic, hence the references must
be taken from the same run that writes the files.

Usage:
    python pollutant_generator.py <output directory>
"""

import numpy as np
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from conftest import create_test_config, test_cases
from network_definition import NetworkDefinition
from pypsa2smspp.transformation import Transformation
from pypsa2smspp.network_correction import (add_slack_unit,
                                            bound_extendable_assets,
                                            clean_ciclicity_storage)


# =============================================================================
# INPUT PARAMETERS
# =============================================================================

# name: (Excel case, {carrier attribute: rate by carrier},
#        [(GlobalConstraint type, carrier attribute, sense, fraction of the
#          value of the unconstrained dispatch)])
# the seed of the network, the same one instance_generator.py uses, so that
# these instances and the plain one of the same case are the same network
SEED = 20260918

VARIANTS = {
    "co2_200": ("3n_3c_1gext_1h_1bext_2l", {"co2_emissions": {"CCGT": 0.35}},
                [("primary_energy", "co2_emissions", "<=", 2.0)]),
    "co2_50": ("3n_3c_1gext_1h_1bext_2l", {"co2_emissions": {"CCGT": 0.35}},
               [("primary_energy", "co2_emissions", "<=", 0.5)]),
    "co2_nox": ("3n_3c_1gext_1h_1bext_2l",
                {"co2_emissions": {"CCGT": 0.35}, "nox_emissions": {"CCGT": 0.001}},
                [("primary_energy", "co2_emissions", "<=", 0.6),
                 ("primary_energy", "nox_emissions", "<=", 0.7)]),
    "co2_min": ("3n_3c_1gext_1h_1bext_2l", {"co2_emissions": {"CCGT": 0.35}},
                [("primary_energy", "co2_emissions", ">=", 1.1)]),
    "ccgt_limit": ("3n_3c_1gext_1h_1bext_2l", {},
                   [("operational_limit", "CCGT", "<=", 0.5)]),
    "co2_h2": ("3n_3c_1gext_1h_1bext_2l",
               {"co2_emissions": {"CCGT": 0.35, "H2": -0.1}},
               [("primary_energy", "co2_emissions", "<=", 0.5)]),
    "co2_eq": ("3n_3c_1gext_1h_1bext_2l", {"co2_emissions": {"CCGT": 0.35}},
               [("primary_energy", "co2_emissions", "==", 0.8)]),
    "co2_hydro": ("3n_3c_1gext_1h_1bext_2l",
                  {"co2_emissions": {"CCGT": 0.35, "hydro": 0.1}},
                  [("primary_energy", "co2_emissions", "<=", 0.5)]),
}

SOLVER_NAME = "highs"


# =============================================================================
# FUNCTIONS
# =============================================================================

def dispatch_value(n, gc_type, attribute):
    """The value of the generators of an optimized network that a limit bounds."""
    if gc_type == "operational_limit":
        gens = n.generators.index[n.generators.carrier == attribute]
        return (n.generators_t.p[gens].multiply(n.snapshot_weightings.generators,
                                                axis=0)).sum().sum()
    return emissions(n, attribute)


def emissions(n, attribute):
    """Emissions of the dispatch of an optimized network, generators only."""
    weights = n.snapshot_weightings.generators
    efficiency = n.get_switchable_as_dense("Generator", "efficiency")
    total = 0.0
    for gen, carrier in n.generators.carrier.items():
        rate = n.carriers.at[carrier, attribute]
        if rate:
            total += rate * (n.generators_t.p[gen] / efficiency[gen] * weights).sum()
    return total


def design_bounds_mode():
    """The way the design bounds are written, from SMSPP_DESIGN_BOUNDS."""
    return os.environ.get("SMSPP_DESIGN_BOUNDS", "physical")


def generate(name, case, rates, limits, out_dir):
    """Write the instance of one variant and return its reference objective."""
    paths = {p.stem: p for p in test_cases["xlsx_paths"]}
    np.random.seed(SEED)
    n = NetworkDefinition(create_test_config(paths[case])).n
    n = clean_ciclicity_storage(n)
    n = add_slack_unit(n)
    n = bound_extendable_assets(n, design_bounds_mode())

    for attribute, by_carrier in rates.items():
        if attribute not in n.carriers.columns:
            n.carriers[attribute] = 0.0
        for carrier, rate in by_carrier.items():
            n.carriers.at[carrier, attribute] = rate

    free = n.copy()
    free.optimize(solver_name=SOLVER_NAME)
    for i, (gc_type, attribute, sense, fraction) in enumerate(limits):
        n.add("GlobalConstraint", f"limit_{i}", type=gc_type,
              carrier_attribute=attribute, sense=sense,
              constant=fraction * dispatch_value(free, gc_type, attribute))

    network = n.copy()
    network.optimize(solver_name=SOLVER_NAME)
    obj_pypsa = float(network.objective + getattr(network, "objective_constant", 0.0))

    file_name = f"smspp_{case}_{name}"
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    transformation = Transformation(
        capacity_expansion_ucblock=True,
        workdir=work_dir,
        name=file_name,
        overwrite=True,
        fp_temp="{name}.nc",
        fp_log="{name}_log.txt",
        fp_solution="{name}_solution.nc",
        configfile="auto",
        pysmspp_options={},
    )
    transformation.run(network, verbose=False)
    shutil.copy(work_dir / f"{file_name}.nc", out_dir / f"{file_name}.nc")

    return f"{file_name}.nc", obj_pypsa, float(transformation.result.objective_value)


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (case, rates, limits) in VARIANTS.items():
        file_name, obj_pypsa, obj_smspp = generate(name, case, rates, limits, out_dir)
        print(f"REF_OBJ[{file_name}]={obj_pypsa:.9e}  # SMS++ {obj_smspp:.9e}")
