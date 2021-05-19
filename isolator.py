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
from functools import partial
import math

from pytools.obj_array import obj_array_vectorize
import pickle

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw, flatten, unflatten
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DTAG_BOUNDARY

from mirgecom.profiling import PyOpenCLProfilingArrayContext

from mirgecom.euler import inviscid_operator
from mirgecom.artificial_viscosity import av_operator
from mirgecom.simutil import (
    inviscid_sim_timestep,
    sim_checkpoint,
    check_step,
    create_parallel_grid
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools

from mirgecom.integrators import rk4_step, euler_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedBoundary,
    AdiabaticSlipBoundary,
    AdiabaticNoslipMovingBoundary,
    DummyBoundary
)
from mirgecom.initializers import (
    Lump
)
from mirgecom.eos import IdealSingleGas

logger = logging.getLogger(__name__)

def get_mesh():

    from meshmode.mesh.io import (
        read_gmsh,
        generate_gmsh,
        ScriptWithFilesSource
    )

    meshfile="/usr/workspace/hagen7/mirgecom/lassen/mergetesting/ns/emirge/mirgecom/examples/isolator/isolator_0p5_diverge.msh"
    mesh = read_gmsh(meshfile,force_ambient_dim=2)

    return mesh

def mass_source(discr,q,r,eos,t,rate):

     from pytools.obj_array import flat_obj_array, make_obj_array
     from mirgecom.euler import split_conserved
     from mirgecom.initializers import _make_pulse
     dim = discr.dim
     cv = split_conserved(dim,q)
     zeros = 0*r[0]
     #actx = r[0].array_context
     r0 = np.zeros(dim)
     r0[0] = 0.68;
     r0[1] = -0.02
     rho_addition = _make_pulse(rate,r0,0.001,r)
     gamma = 1.289
     R = 8314.59/44.009
     temp_inflow=297.169
     e = temp_inflow*R/(gamma-1.)
     rhoe_addition = rho_addition*e
     return flat_obj_array(rho_addition,rhoe_addition,zeros,zeros)
     
def sponge(q,q_ref,sigma):
    return(sigma*(q_ref-q))


class Discontinuity:
    r"""Initializes the flow to a discontinuous state, planar located at x=xloc
    The inital condition is defined
    .. math::
        (\rho,u,P) = 
    This function only serves as an initial condition
    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self,dim=2, x0=0., rhol=0.1, rhor=0.01, pl=20, pr=10., ul=0.1, ur=0., sigma=0.5
    ):
        """Initialize initial condition options
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

    def __call__(self, t, x_vec, eos=IdealSingleGas()):
        """
        Create the discontinuity at locations *x_vec*.
        profile is defined by <left_val>/2.0*(tanh(-(x-x0)/\sigma)+1)+<right_val>/2.0*(tanh((x-x0)/\sigma)+1.0)
        Parameters
        ----------
        t: float
            Current time at which the solution is desired (unused)
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :classg
s.GasEO      Stion of state class to be used in construction of soln (if needed)
        """
        x_rel = x_vec[0]
        actx = x_rel.array_context
        gm1 = eos.gamma() - 1.0
        zeros = 0*x_rel
        sigma=self._sigma

        x0 = zeros + self._x0
        t = zeros + t
        
        rhol = zeros + self._rhol
        rhor = zeros + self._rhor
        ul = zeros + self._ul
        ur = zeros + self._ur
        rhoel = zeros + self._pl/gm1
        rhoer = zeros + self._pr/gm1

        xtanh = 1.0/sigma*(x_rel-x0)
        mass = rhol/2.0*(actx.np.tanh(-xtanh)+1.0)+rhor/2.0*(actx.np.tanh(xtanh)+1.0)
        rhoe = rhoel/2.0*(actx.np.tanh(-xtanh)+1.0)+rhoer/2.0*(actx.np.tanh(xtanh)+1.0)
        u = ul/2.0*(actx.np.tanh(-xtanh)+1.0)+ur/2.0*(actx.np.tanh(xtanh)+1.0)
        rhou = mass*u
        energy = rhoe + 0.5*mass*(u*u)

        from pytools.obj_array import make_obj_array
        from mirgecom.fluid import join_conserved
        mom = make_obj_array(
            [
                0*x_rel
                for i in range(self._dim)
            ]
        )
        mom[0]=rhou

        return join_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom)

@mpi_entry_point
def main(ctx_factory=cl.create_some_context,
         snapshot_pattern="isolator-{step:06d}-{rank:04d}.pkl",
         restart_step=None, use_profiling=False, use_logmgr=False):
    """Drive the Y0 example."""

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    """logging and profiling"""

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))


    nrestart = 5000
    nviz = 500
    current_dt = 1e-7
    t_final = 1.e0

    dim = 2
    order = 3
    exittol = .09
    current_cfl = 1.0
    vel_init = np.zeros(shape=(dim,))
    vel_inflow = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    orig[0] = 0.83
    current_dt = 0.1e-7
    current_t = 0
    casename = "isolator"
    constant_cfl = False
    # no internal euler status messages
    nstatus = 25
    checkpoint_t = current_t
    current_step = 0
    so = np.log10(1.e-4/np.power(order,4))
    epsilon = 3.0e-1*(1.0)/order #This is using a porportionality constant of 3.0e-1 and taking 0p5x as the base grid h
    print(so, epsilon, flush=True)

    # working gas: CO2 #
    #   gamma = 1.289
    #   MW=44.009  g/mol
    #   cp = 37.135 J/mol-K,
    #   rho= 1.977 kg/m^3 @298K
    gamma_CO2 = 1.289
    R_CO2 = 8314.59/44.009

    # background
    #   100 Pa
    #   298 K
    #   rho = 1.77619667e-3 kg/m^3
    #   velocity = 0,0,0
    rho_bkrnd=1.0e-1 #1.77619667e-1
    pres_bkrnd=5000 #10000
    temp_bkrnd=298
     
    # nozzle inflow #
    # 
    # stagnation tempertuare 298 K
    # stagnation pressure 1.5e Pa
    # 
    # isentropic expansion based on the area ratios between the inlet (r=13e-3m) and the throat (r=6.3e-3)
    #
    #  MJA, this is calculated offline, add some code to do it for us
    # 
    #   Mach number=0.139145
    #   pressure=148142
    #   temperature=297.169
    #   density=2.63872
    #   gamma=1.289
    pres_inflow=148142
    temp_inflow=297.169
    rho_inflow=2.63872
    mach_inflow=infloM = 0.139145
    vel_inflow[0] = mach_inflow*math.sqrt(gamma_CO2*pres_inflow/rho_inflow)

    timestepper = euler_step
    eos = IdealSingleGas(gamma=gamma_CO2, gas_const=R_CO2)
    bulk_init = Discontinuity(dim=dim, x0=0.235,sigma=0.004,
                              rhol=rho_inflow, rhor=rho_bkrnd,
                              pl=pres_inflow, pr=pres_bkrnd,
                              ul=vel_inflow[0], ur=300.)
    inflow_init = Lump(dim=dim, rho0=rho_inflow, p0=pres_inflow,
                       center=orig, velocity=vel_inflow, rhoamp=0.0)
    wall = AdiabaticNoslipMovingBoundary()
    dummy = DummyBoundary()

    boundaries = {DTAG_BOUNDARY("inflow"): PrescribedBoundary(inflow_init),
                  DTAG_BOUNDARY("outflow"): dummy,
                  DTAG_BOUNDARY("wall"): wall}


    

    if restart_step is None:
        local_mesh, global_nelements = create_parallel_grid(comm, get_mesh)
        local_nelements = local_mesh.nelements
    else:  # Restart
        with open(snapshot_pattern.format(step=restart_step, rank=rank), "rb") as f:
            restart_data = pickle.load(f)

        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]

        assert comm.Get_size() == restart_data["num_parts"]

    if rank == 0:
        logging.info("Making discretization")
    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())

    zeros = 0.0*nodes[0]
    from pytools.obj_array import flat_obj_array, make_obj_array
    state_init = flat_obj_array(0.15+zeros,31500.+zeros,513*0.15+zeros,zeros)
    sigma =  1.0e-2/current_dt * 0.5*(1.+ actx.np.tanh((nodes[0]-.95)/0.02))
    if restart_step is None:
        if rank == 0:
            logging.info("Initializing soln.")
        current_state = bulk_init(0, nodes, eos=eos)
    else:
        current_t = restart_data["t"]
        current_step = restart_step

        current_state = unflatten(
            actx, discr.discr_from_dd("vol"),
            obj_array_vectorize(actx.from_numpy, restart_data["state"]))

    vis_timer = None

    visualizer = make_visualizer(discr, discr.order
                                 if discr.dim == 2 else discr.order)

    #    initname = initializer.__class__.__name__
    initname = "isolator"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    get_timestep = partial(inviscid_sim_timestep, discr=discr, t=current_t,
                           dt=current_dt, cfl=current_cfl, eos=eos,
                           t_final=t_final, constant_cfl=constant_cfl)

    def my_rhs(t, state):
        nanval = state[0][0][0][0]
        if nanval != nanval:
            exit()
        return ( 
                 inviscid_operator(discr, q=state, t=t,boundaries=boundaries, eos=eos)
               + av_operator(discr, q=state, boundaries=boundaries, boundary_kwargs={"time": t, "eos": eos}, alpha=epsilon,s0=so)
               #+ mass_source(discr,t=t,q=state,eos=eos,rate=0.25e5,r=nodes)
               + sponge(q=state,q_ref=state_init,sigma=sigma)
               )

    def my_checkpoint(step, t, dt, state):

        write_restart = (check_step(step, nrestart)
                         if step != restart_step else False)
        if write_restart is True:
            with open(snapshot_pattern.format(step=step, rank=rank), "wb") as f:
                pickle.dump({
                    "local_mesh": local_mesh,
                    "state": obj_array_vectorize(actx.to_numpy, flatten(state)),
                    "t": t,
                    "step": step,
                    "global_nelements": global_nelements,
                    "num_parts": nparts,
                    }, f)

        return sim_checkpoint(discr=discr, visualizer=visualizer, eos=eos,
                              q=state, vizname=casename,
                              step=step, t=t, dt=dt, nstatus=nstatus,
                              nviz=nviz, exittol=exittol,
                              constant_cfl=constant_cfl, comm=comm,
                              overwrite=True)

    if rank == 0:
        logging.info("Stepping.")

    # Set restart step so the advance routine doesn't bug
    if restart_step is None:
        restart_step=0

    (current_step, current_t, current_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      checkpoint=my_checkpoint, istep=restart_step,
                      get_timestep=get_timestep, state=current_state,
                      t_final=t_final, t=current_t)

    if rank == 0:
        logger.info("Checkpointing final state ...")

    my_checkpoint(current_step, t=current_t,
                  dt=(current_t - checkpoint_t),
                  state=current_state)

    if current_t - t_final < 0:
        raise ValueError("Simulation exited abnormally")


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    use_logging = False
    use_profiling = False

    # crude command line interface
    # get the restart interval from the command line
    print(f"Running {sys.argv[0]}\n")
    nargs = len(sys.argv)
    if nargs > 1:
        restart_step = int(sys.argv[1])
        print(f"Restarting from step {restart_step}")
        main(restart_step=restart_step,use_profiling=use_profiling,use_logmgr=use_logging)
    else:
        print(f"Starting from step 0")
        main(use_profiling=use_profiling,use_logmgr=use_logging)
    #main()

# vim: foldmethod=marker
