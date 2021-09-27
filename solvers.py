from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, linalg
import meshing

@dataclass
class Fluid:
    rho: float
    nu: float

def add_coeff(mdict, i, j, v):
    if (i, j) not in mdict:
        mdict[(i, j)] = 0
    mdict[(i, j)] += v

def dict_to_coo(mdict, n):
    i = [k[0] for k in mdict]
    j = [k[1] for k in mdict]
    v = [mdict[k] for k in mdict]
    return coo_matrix((v, (i,j)), shape=(n, n))

def face_interpolate(mesh: meshing.Mesh, vID, vfID, u_prev, u_factor=0.75):
    face = mesh.volumes[vID].face_objs[vfID]
    central = np.array([1-face.face_dist/face.delta, face.face_dist/face.delta])
    umag = np.sqrt(np.sum(u_prev[vID]**2))
    out = 0.5+np.dot(u_prev[vID], face.norm)/(2*umag) if umag != 0 else 1
    upwind = np.array([out, 1-out])
    return (1-u_factor)*central+u_factor*upwind

def run_interpolate(v0, v1, inter):
    return v0*inter[0]+v1*inter[1]

def solve_momentum_eqns(mesh: meshing.Mesh, grad_p, u_prev, fluid: Fluid):
    mdict = {}
    b = np.zeros((mesh.vcount, 2))
    point_vel = np.zeros((mesh.pcount, 2))
    for pID in range(mesh.pcount):
        point = mesh.points[pID]
        vol_weights = np.array([np.sqrt(np.sum((point.pos-mesh.volumes[vID].pos)**2)) for vID in point.volumes])
        vol_weights = np.transpose(np.vstack((vol_weights, vol_weights)))
        point_vel[pID] = np.sum(vol_weights*np.array([u_prev[vID] for vID in point.volumes]))/np.sum(vol_weights)
    for vID in range(mesh.vcount):
        volume = mesh.volumes[vID]
        for vfID in range(len(volume.faces)):
            face = volume.face_objs[vfID]
            # convective
            face_inter = face_interpolate(mesh, vID, vfID, u_prev)
            ufi = run_interpolate(u_prev[vID], u_prev[face.neighbor], face_inter)
            add_coeff(mdict, vID, vID, face_inter[0]*face.area*np.dot(ufi, face.norm))
            add_coeff(mdict, vID, face.neighbor, face_inter[1]*face.area*np.dot(ufi, face.norm))
            # diffusive
            add_coeff(mdict, vID, vID, -fluid.nu*face.area/face.delta)
            add_coeff(mdict, vID, face.neighbor, fluid.nu*face.area/face.delta)
            uno = (point_vel[face.points[0]]-point_vel[face.points[1]])/face.delta
            b[vID] += fluid.nu*uno*np.dot(face.tan, face.lvec)/face.delta
        b[vID] -= grad_p[vID]*volume.vol/fluid.rho
    for k in mdict:
        mdict[k] = mdict[k]*(1+1j)
    cb = np.array([b[vID,0]+1j*b[vID,1] for vID in range(mesh.vcount)])
    U = linalg.spsolve(dict_to_coo(mdict, mesh.vcount), cb)
    print(U)