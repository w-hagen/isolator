"""mirgecom driver for the Y0 demonstration.

Note: this example requires a *scaled* version of the Y0
grid. A working grid example is located here:
github.com:/illinois-ceesd/data@y0scaled
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import logging
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
import math


from meshmode.array_context import PyOpenCLArrayContext
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DTAG_BOUNDARY
from logpyle import set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_device_name,
    logmgr_add_device_memory_usage,
    set_sim_state
)

from mirgecom.navierstokes import ns_operator
from mirgecom.artificial_viscosity import av_operator
from mirgecom.simutil import (
    get_sim_timestep,
    check_step
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools

from mirgecom.fluid import make_conserved
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedInviscidBoundary,
    AdiabaticNoslipMovingBoundary,
    DummyBoundary
)
from mirgecom.initializers import Lump
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


def get_mesh():
    """Get the mesh."""
    from meshmode.mesh.io import read_gmsh
    mesh_filename = "data/isolator.msh"
    mesh = read_gmsh(mesh_filename, force_ambient_dim=2)
    return mesh


def mass_source(discr, q, r, eos, t, rate):
    """Compute the mass source term."""
    from pytools.obj_array import flat_obj_array
    from mirgecom.initializers import _make_pulse

    dim = discr.dim
    zeros = 0 * r[0]
    r0 = np.zeros(dim)
    r0[0] = 0.68
    r0[1] = -0.02
    rho_addition = _make_pulse(rate, r0, 0.001, r)
    gamma = 1.289
    r_gas = 8314.59 / 44.009
    temp_inflow = 297.169
    e = temp_inflow * r_gas / (gamma - 1.0)
    rhoe_addition = rho_addition * e
    return flat_obj_array(rho_addition, rhoe_addition, zeros, zeros)


def sponge(q, q_ref, sigma):
    """Compute specific sponge term."""
    return sigma * (q_ref - q)


class Discontinuity:
    r"""Initializes the flow to a discontinuous state, planar located at x=*xloc*.

    The inital condition is defined
    .. math::
        (\rho,u,P) =

    This function only serves as an initial condition

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, dim=2, x0=0.0, rhol=0.1, rhor=0.01, pl=20, pr=10.0, ul=0.1,
                 ur=0.0, sigma=0.5):
        """Initialize initial condition options.

        Parameters
        ----------
        dim: int
           dimension of domain
        x0: float
           location of discontinuity
        rhol: float
           left density
        rhor: float
           right density
        pl: float
           left pressure
        pr: float
           right pressure
        ul: float
           left velocity
        ur: float
           right velocity
        sigma: float
           sharpness parameter
        """
        self._dim = dim
        self._x0 = x0
        self._rhol = rhol
        self._rhor = rhor
        self._pl = pl
        self._pr = pr
        self._ul = ul
        self._ur = ur
        self._sigma = sigma

    def __call__(self, x_vec, *, eos=None, t=None):
        r"""Create the discontinuity at locations *x_vec*.

        The profile is defined by
        $<left_val>/2.0*(tanh(-(x-x0)/\sigma)+1)+<right_val>/2.0*(tanh((x-x0)/\sigma)+1.0)$.

        Parameters
        ----------
        t: float
            Current time at which the solution is desired (unused)
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`~mirgecom.eos.GasEOS`
            Equation of state class to be used in construction of soln (if needed)
        """
        x_rel = x_vec[0]
        actx = x_rel.array_context
        gm1 = eos.gamma() - 1.0
        zeros = 0 * x_rel
        sigma = self._sigma

        x0 = zeros + self._x0
        t = zeros + t

        rhol = zeros + self._rhol
        rhor = zeros + self._rhor
        ul = zeros + self._ul
        ur = zeros + self._ur
        rhoel = zeros + self._pl / gm1
        rhoer = zeros + self._pr / gm1

        xtanh = 1.0 / sigma * (x_rel - x0)
        mass = rhol / 2.0 * (actx.np.tanh(-xtanh) + 1.0) + rhor / 2.0 * (
            actx.np.tanh(xtanh) + 1.0
        )
        rhoe = rhoel / 2.0 * (actx.np.tanh(-xtanh) + 1.0) + rhoer / 2.0 * (
            actx.np.tanh(xtanh) + 1.0
        )
        u = ul / 2.0 * (actx.np.tanh(-xtanh) + 1.0) + ur / 2.0 * (
            actx.np.tanh(xtanh) + 1.0
        )
        rhou = mass * u
        energy = rhoe + 0.5 * mass * (u * u)

        from pytools.obj_array import make_obj_array
        mom = make_obj_array([0 * x_rel for i in range(self._dim)])
        mom[0] = rhou

        return make_conserved(dim=self._dim, mass=mass, energy=energy, momentum=mom)


@mpi_entry_point
def main(
        ctx_factory=cl.create_some_context, rst_filename=None, use_profiling=False,
        use_logmgr=False, casename="isolator"):
    """Drive the Y0 example."""
    cl_ctx = ctx_factory()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    """logging and profiling"""

    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            logmgr=logmgr)
    else:
        queue = cl.CommandQueue(cl_ctx)
        actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    current_dt = 1e-8
    t_final = 1e-7
    current_t = 0
    current_step = 0
    current_cfl = 1.0
    constant_cfl = False

    # no internal euler status messages
    nviz = 500
    nhealth = 1
    nrestart = 5000
    nstatus = 25

    order = 3
    so = np.log10(1.0e-4 / np.power(order, 4))
    epsilon = (
        3.0e-1 * (1.0) / order
    )
    # This uses proportionality constant of 3.0e-1 and taking 0p5x as the base grid h
    print(so, epsilon, flush=True)

    # {{{ Initialize simple transport model
    kappa = 1.0e-5
    sigma = 1.0e-5
    transport_model = SimpleTransport(viscosity=sigma, thermal_conductivity=kappa)
    # }}}
    # working gas: CO2 #
    #   gamma = 1.289
    #   MW=44.009  g/mol
    #   cp = 37.135 J/mol-K,
    #   rho= 1.977 kg/m^3 @298K
    gamma_co2 = 1.289
    r_co2 = 8314.59 / 44.009

    # background
    #   100 Pa
    #   298 K
    #   rho = 1.77619667e-3 kg/m^3
    #   velocity = 0,0,0
    rho_bkrnd = 1.0e-1  # 1.77619667e-1
    pres_bkrnd = 5000  # 10000

    # nozzle inflow #
    #
    # stagnation tempertuare 298 K
    # stagnation pressure 1.5e Pa
    #
    # isentropic expansion based on the area ratios between the inlet (r=13e-3m) and
    # the throat (r=6.3e-3)
    #
    #  MJA, this is calculated offline, add some code to do it for us
    #
    #   Mach number=0.139145
    #   pressure=148142
    #   temperature=297.169
    #   density=2.63872
    #   gamma=1.289
    dim = 2
    vel_inflow = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    orig[0] = 0.83
    pres_inflow = 148142
    rho_inflow = 2.63872
    mach_inflow = 0.139145
    vel_inflow[0] = mach_inflow * math.sqrt(gamma_co2 * pres_inflow / rho_inflow)

    from mirgecom.integrators import euler_step
    timestepper = euler_step
    eos = IdealSingleGas(
        gamma=gamma_co2, gas_const=r_co2, transport_model=transport_model
    )
    bulk_init = Discontinuity(dim=dim, x0=0.235, sigma=0.004, rhol=rho_inflow,
                              rhor=rho_bkrnd, pl=pres_inflow, pr=pres_bkrnd,
                              ul=vel_inflow[0], ur=300.0)
    inflow_init = Lump(dim=dim, rho0=rho_inflow, p0=pres_inflow, center=orig,
                       velocity=vel_inflow, rhoamp=0.0)
    wall = AdiabaticNoslipMovingBoundary()
    dummy = DummyBoundary()

    boundaries = {
        DTAG_BOUNDARY("inflow"):
        PrescribedInviscidBoundary(fluid_solution_func=inflow_init),
        DTAG_BOUNDARY("outflow"): dummy,
        DTAG_BOUNDARY("wall"): wall,
    }

    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    if rst_filename:  # read the grid from restart data
        rst_filename = f"{rst_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["nparts"] == nparts
    else:  # generate the grid from scratch
        from mirgecom.simutil import generate_and_distribute_mesh
        local_mesh, global_nelements = generate_and_distribute_mesh(comm, get_mesh)
        local_nelements = local_mesh.nelements

    if rank == 0:
        logging.info("Making discretization")
    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    from meshmode.dof_array import thaw
    nodes = thaw(actx, discr.nodes())

    zeros = discr.zeros(actx)
    from pytools.obj_array import make_obj_array
    mom_init = make_obj_array([zeros + 513 * 0.15, zeros]) 
    state_init = make_conserved(dim, mass=0.15+zeros, energy=315000.0+zeros,
                                momentum=mom_init)
    sigma = (
        1.0e-2 / current_dt * 0.5 * (1.0 + actx.np.tanh((nodes[0] - 0.95) / 0.02))
    )

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, discr, dim,
                             extract_vars_for_logging, units_for_logging)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure",    "{value:1.9e})\n"),
            ("min_temperature", "------- T (min, max) (K) = ({value:1.9e}, "),
            ("max_temperature",    "{value:1.9e})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_state = restart_data["state"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        if rank == 0:
            logging.info("Initializing soln.")
        current_state = bulk_init(nodes, eos=eos, t=0)

    visualizer = make_visualizer(
        discr, discr.order if discr.dim == 2 else discr.order
    )

    #    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order, nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=casename,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    def my_write_viz(step, t, state, dv=None, tagged_cells=None):
        if dv is None:
            dv = eos.dependent_vars(state)
        if tagged_cells is None:
            from mirgecom.artificial_viscosity import smoothness_indicator
            tagged_cells = smoothness_indicator(discr, state.mass, s0=so,
                                                kappa=kappa)
        viz_fields = [("cv", state),
                      ("dv", dv),
                      ("tagged_cells", tagged_cells)]
        from mirgecom.simutil import write_visfile
        write_visfile(discr, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "state": state,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(pressure):
        # Note: This health check is tuned s.t. it is a test that
        #       the case gets the expected solution.  If dt,t_final or
        #       other run parameters are changed, this check should
        #       be changed accordingly.
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        from mirgecom.simutil import allsync
        if allsync(check_range_local(discr, "vol", pressure, 1000, 1.5e5),
                   comm, op=MPI.LOR):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            p_min = nodal_min(discr, "vol", pressure)
            p_max = nodal_max(discr, "vol", pressure)
            logger.info(f"Pressure range violation ({p_min=}, {p_max=})")

        return health_error

    def my_pre_step(step, t, dt, state):
        try:
            dv = None

            if logmgr:
                logmgr.tick_before()

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                dv = eos.dependent_vars(state)
                from mirgecom.simutil import allsync
                health_errors = allsync(my_health_check(dv.pressure), comm,
                                        op=MPI.LOR)
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                if dv is None:
                    dv = eos.dependent_vars(state)
                from mirgecom.artificial_viscosity import smoothness_indicator
                tagged_cells = smoothness_indicator(discr, state.mass, s0=so,
                                                    kappa=kappa)
                my_write_viz(step=step, t=t, state=state, dv=dv,
                             tagged_cells=tagged_cells)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=state)
            my_write_restart(step=step, t=t, state=state)
            raise

        dt = get_sim_timestep(discr, state, t, dt, current_cfl, eos, t_final,
                              constant_cfl)

        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        return ns_operator(
            discr, cv=state, t=t, boundaries=boundaries, eos=eos
        ) + make_conserved(dim, q=av_operator(
            discr, q=state.join(), boundaries=boundaries,
            boundary_kwargs={"time": t, "eos": eos}, alpha=epsilon,
            s0=so, kappa=kappa)
        ) + make_conserved(dim, q=sponge(q=state.join(), q_ref=state_init.join(),
                                         sigma=sigma))

    current_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                  current_cfl, eos, t_final, constant_cfl)

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_state, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = eos.dependent_vars(current_state)
    my_write_viz(step=current_step, t=current_t, state=current_state, dv=final_dv)
    my_write_restart(step=current_step, t=current_t, state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import sys

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    use_logging = True
    use_profiling = False

    # crude command line interface
    # get the restart interval from the command line
    print(f"Running {sys.argv[0]}\n")
    nargs = len(sys.argv)
    if nargs > 1:
        rst_filename = sys.argv[1]
        print(f"Restarting from {rst_filename=}.")
        main(
            rst_filename=rst_filename,
            use_profiling=use_profiling,
            use_logmgr=use_logging,
        )
    else:
        print("Starting from step 0.")
        main(use_profiling=use_profiling, use_logmgr=use_logging)

# vim: foldmethod=marker
