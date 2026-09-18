# -*- coding: utf-8 -*-
"""
Kirchhoff's voltage law, i.e., the susceptance of the lines.

PyPSA bounds the flow of a `Line` by its reactance, which the conversion writes
as the susceptance of the network when it is asked to, i.e., with
`kirchhoff_voltage_law`; with it off the AC network is a transport one, as it
has always been. Here the susceptance written in the netCDF file is read back,
and so is the BlockConfig the conversion writes next to the instance, which
asks for the KIRCHHOFF formulation: the one the solver takes by default is not
equivalent to it on a network of AC lines and HVDC links up to UCBlock
2b69e107. None of this needs SMS++; that the two objectives agree is checked in
`test_ucblock.py`.
"""
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pypsa

from conftest import safe_remove, OUT_TEST

from pypsa2smspp.transformation import Transformation
from pypsa2smspp.utils import write_kirchhoff_config


SNAPSHOTS = 6


def two_bus_network() -> pypsa.Network:
    """Two buses joined by a line and by a link, one generator on each."""
    n = pypsa.Network()
    n.set_snapshots(range(SNAPSHOTS))
    n.add("Bus", ["bus0", "bus1"], v_nom=380.0)
    n.add("Carrier", ["ocgt", "slack"])
    n.add("Load", "load", bus="bus1", p_set=np.full(SNAPSHOTS, 500.0))
    n.add("Line", "line", bus0="bus0", bus1="bus1", x=20.0, r=0.1, s_nom=800.0)
    n.add("Link", "link", bus0="bus0", bus1="bus1", p_nom=200.0)
    n.add("Generator", "cheap", bus="bus0", carrier="ocgt", p_nom=2000.0,
          marginal_cost=10.0)
    n.add("Generator", "shedding", bus="bus1", carrier="slack", p_nom=1e5,
          marginal_cost=3000.0)
    return n


def susceptance(case_name, **options):
    """The susceptances the conversion writes for the two-bus network."""
    temp_nc = OUT_TEST / f"kirchhoff_{case_name}.nc"
    safe_remove(temp_nc)

    network = two_bus_network()
    transformation = Transformation(capacity_expansion_ucblock=True, **options)
    transformation.create_model(network, verbose=False)
    transformation.sms_network.to_netcdf(temp_nc, force=True)

    network.calculate_dependent_values()
    with nc.Dataset(temp_nc) as dataset:
        return network, np.ravel(dataset["Block_0"]["LineSusceptance"][...])


def test_susceptance_is_the_inverse_reactance():
    network, values = susceptance("on", kirchhoff_voltage_law=True)

    # the line first, the link after it, the latter having no susceptance
    assert values[0] == 1.0 / network.lines.at["line", "x_pu_eff"]
    assert values[1] == 0.0


def test_the_network_is_a_transport_one_by_default():
    _, values = susceptance("off")

    assert not np.any(values)


def test_the_block_config_asks_for_kirchhoff():
    directory = OUT_TEST / "kirchhoff_config"
    directory.mkdir(parents=True, exist_ok=True)

    meta = Path(write_kirchhoff_config(directory, "case"))
    assert meta.is_absolute()

    text = meta.read_text()
    assert "DCNetworkBlock" in text

    # the file the map points at is named by an absolute path, the solver
    # resolving a relative one against a prefix of its own
    referred = Path(text.split("*")[-1].strip())
    assert referred.is_absolute() and referred.exists()
    assert "SimpleConfiguration<int>" in referred.read_text()
