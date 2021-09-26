import meshing
import objects
import postprocess

if __name__ == "__main__":
    airfoil = objects.create_naca_airfoil(4412, 0.2, 640, alpha=5*3.14159/180)
    mesh = meshing.Mesh(airfoil)
    mesh.generate(0.6, 0.25, layers=50)

    postprocess.add_scalarfield(mesh, [vol.vol for vol in mesh.volumes])
    postprocess.add_mesh(mesh)
    postprocess.set_limits(mesh)
    postprocess.show()