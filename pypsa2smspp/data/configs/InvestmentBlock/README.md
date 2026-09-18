# The configuration of an InvestmentBlock

What `Transformation` hands to SMS++ when it writes an InvestmentBlock and
`configfile` is left at "auto". `BSPar.txt` is the one that is used, and it is
the configuration the released SMS++ reads, i.e. the one whose bundle is the
1.0 and whose master is QPPenalty.

A configuration is read as a whole: a parameter the SMS++ at hand does not
know makes its whole ComputeConfig fail to load, and it is dropped in silence,
taking with it the parameters that would have been read. `strInnerBSC` is one
of those, so the inner Block is left to the `BSCfg.txt` of this folder, which
is the name the tool looks for anyway.

Two things these instances take are therefore written here but not set:

- the master of the bundle: a feasibility cut reaches it as a constraint,
  which QPPenalty refuses outright, and on an instance where the design of an
  asset sits at 0 QPPenalty also stops on a point four times the optimum. The
  master that takes constraints is the OSI one, `intMPName 15`, and a build
  without Osi has none, the released SMS++ among them;
- the unbounded dual direction a feasibility cut is read off, which takes
  `intHomogeneousDirection 1` in `BSCfg.txt` and a Solver that returns one,
  i.e. CPLEX or GUROBI. It is left on HiGHS, which every build has, so an
  instance whose inner Block the design can starve, a sector-coupled network
  with an extendable generator for one, ends with no answer until that line is
  uncommented.

`BSPar_2.0.txt` is the configuration for the BundleSolver 2.0, whose master is
a MasterProblemBlock solved by a :MILPSolver (`strMPBSolverCfg` ->
`MPBCfg.txt`, `strInnerBSC` -> `BSCfg1.txt` on GUROBI). It solves every
instance of the set, both the one where a design sits at 0 and the one whose
inner Block the design starves, and `test/instance_generator.py` passes it
explicitly, being run against a build of the 2.0. The presolve of its master
is left at its default: with it off the master of a design over several
extendable lines ends in "Bundle::FormD: unrecoverable MP failure".

The same `BSPar_2.0.txt`, `MPBCfg.txt` and `BSCfg1.txt` serve an InvestmentBlock
whose inner Block is a whole TwoStageStochasticBlock or MultiStageStochasticBlock,
as `investment_outside` writes it, i.e., a Benders decomposition with the design
in the master and the scenarios, solved together by GUROBI, in the value
function: `strInnerBSC` gives `BSCfg1.txt` to the stochastic Block, and its
`intHomogeneousDirection 1` is what lets a design that makes a scenario
infeasible produce a feasibility cut. On the stochastic instances of the tests
this form reaches the value of the deterministic equivalent, but it is not
faster than it, as expected with a single block of complicating variables; it
has not yet been measured with these files, whose master differs from the one
of those runs.
