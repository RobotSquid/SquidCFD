import meshing
import objects
import solvers
import postprocess
import numpy as np

if __name__ == "__main__":
    airfoil = objects.create_naca_airfoil(4412, 0.2, 640, alpha=5*3.14159/180)
    mesh = meshing.Mesh(airfoil)
    mesh.generate(0.6, 0.25, layers=50)

    #postprocess.add_scalarfield(mesh, [vol.vol for vol in mesh.volumes])
    #postprocess.add_mesh(mesh)
    #postprocess.set_limits(mesh)
    #postprocess.show()

    air_stp = solvers.Fluid(1.225, 1.48e-5)
    grad_p = np.zeros((mesh.vcount, 2))
    u_prev = np.zeros((mesh.vcount, 2))
    U = solvers.solve_momentum_eqns(mesh, grad_p, u_prev, air_stp)