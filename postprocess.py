import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(subplot_kw={'aspect':'equal'})

def add_scalarfield(mesh, volvalues):
    plt.tripcolor(mesh.raw_points[:,0], mesh.raw_points[:,1], mesh.raw_volumes, volvalues, shading='flat')

def add_mesh(mesh, color='k', linewidth=0.2):
    plt.triplot(mesh.raw_points[:,0], mesh.raw_points[:,1], mesh.raw_volumes, color=color, linewidth=linewidth)

def show():
    plt.show()

def set_limits(mesh):
    plt.xlim((np.min(mesh.raw_points[:,0])*1.1, np.max(mesh.raw_points[:,0])*1.1))
    plt.ylim((np.min(mesh.raw_points[:,1])*1.1, np.max(mesh.raw_points[:,1])*1.1))