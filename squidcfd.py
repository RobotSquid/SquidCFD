import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import meshing
import objects

fig, ax = plt.subplots(subplot_kw={'aspect':'equal'})

def display_mesh(points, tris, values):
    plt.tripcolor(points[:,0], points[:,1], tris, facecolors=values, edgecolors='k')
    plt.xlim((np.min(points[:,0])*1.1, np.max(points[:,0])*1.1))
    plt.ylim((np.min(points[:,1])*1.1, np.max(points[:,1])*1.1))

if __name__ == "__main__":
    block = objects.create_naca_airfoil(4412, 0.2, 640, alpha=np.deg2rad(10))
    mesh = meshing.Mesh(block)
    mesh.generate(0.6, 0.25, layers=50)

    display_mesh(np.array([point.pos for point in mesh.points]), [tri.points for tri in mesh.volumes], np.array([vol.vol for vol in mesh.volumes]))
    
    plt.show()