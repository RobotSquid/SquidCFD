import numpy as np
from scipy import interpolate, spatial
import objects
from dataclasses import dataclass, field

def normal_offset_eval(obj, npts, offset, n=1):
    tck, u = interpolate.splprep(np.transpose(obj), s=0)
    unew = np.linspace(0, 1, npts)
    dirc = interpolate.splev(unew, tck, der=1)
    dircl = np.sqrt(dirc[0]**2 + dirc[1]**2)
    norm = np.transpose(np.array([-dirc[1]/dircl, dirc[0]/dircl]))
    out = np.transpose(interpolate.splev(unew, tck)) + norm*offset
    for i in range(n):
        out[-1] = out[0]
        tck, u = interpolate.splprep(np.transpose(out), s=0, per=len(out))
        out = np.transpose(interpolate.splev(unew, tck))[:-1,:]
    return out

@dataclass
class Point:
    volumes: list[int] = field(default_factory=list)
    pos: np.ndarray = None

@dataclass
class Face:
    norm: np.ndarray = None
    tan: np.ndarray = None
    pos: np.ndarray = None
    area: float = 0
    boundary: bool = False
    points: list[int] = field(default_factory=list)
    volumes: list[int] = field(default_factory=list)

@dataclass
class Volume:
    faces: list[int] = field(default_factory=list)
    flipped: list[bool] = field(default_factory=list)
    neighbors: list[int] = field(default_factory=list)
    points: list[int] = field(default_factory=list)
    vol: float = 0
    pos: np.ndarray = None

class Mesh:
    def __init__(self, obj):
        self.obj = obj
    
    def generate(self, width, height, layers=50, cap=15):
        base_diff = np.diff(self.obj, axis=0)
        base = np.sum(np.sqrt(base_diff[:,0]**2 + base_diff[:,1]**2))/(len(base_diff))
        self.points = self.obj
        last_layer = self.obj
        dist = base*((3/4)**(1/2))
        count = len(self.obj)
        for i in range(layers):
            last_layer = normal_offset_eval(last_layer, int(count), dist, n=(2 if i==0 else 1))
            diffs = np.diff(last_layer, axis=0)
            circ = np.sum(np.sqrt(diffs[:,0]**2+diffs[:,1]**2))
            count = circ/(dist*2/np.sqrt(3))
            if dist < cap*base*((3/4)**(1/2)):
                dist *= 1.1
            self.points = np.concatenate((self.points, last_layer))
        dist /= 2
        self.points = [node for node in self.points if (-width/2+dist)<node[0]<(width/2-dist) and (-height/2+dist)<node[1]<(height/2-dist)]
        self.points = np.concatenate((self.points, objects.create_block(width, height, 4*(int(count)//4))))
        self.volumes = spatial.Delaunay(self.points).simplices
        self.volumes = [vol for vol in self.volumes if not all(vert < len(self.obj) for vert in vol)]
        
        self.points = [Point(pos=node) for node in self.points]
        self.volumes = [Volume(points=vol) for vol in self.volumes]
        self.faces = []
        assigned = {}
        for vID in range(len(self.volumes)):
            volume = self.volumes[vID]
            volume.pos = np.sum([self.points[i].pos for i in volume.points], axis=0)/len(volume.points)
            for i in range(len(volume.points)):
                self.points[volume.points[i]].volumes.append(vID)
                pID1 = volume.points[i]
                pID2 = volume.points[(i+1)%len(volume.points)]
                raw_points = [pID1, pID2]
                volume.vol += np.linalg.det(np.vstack((self.points[pID1].pos, self.points[pID2].pos)))
                face = tuple(sorted(raw_points))
                volume.flipped.append(face in assigned)
                if face not in assigned:
                    assigned[face] = (len(self.faces), [])
                    self.faces.append(Face(points=raw_points))
                fID = assigned[face][0]
                self.faces[fID].volumes.append(vID)
                assigned[face][1].append({vID})
                volume.faces.append(fID)
            volume.vol /= 2
        for face in self.faces:
            face.pos = np.sum([self.points[i].pos for i in face.points], axis=0)/len(face.points)
            face.tan = self.points[face.points[1]].pos-self.points[face.points[0]].pos
            face.area = np.sqrt(np.sum(face.tan**2))
            face.norm = (np.array(((0, 1), (-1, 0))) @ face.tan)/face.area
            face.boundary = len(face.volumes) == 1
        for volume in self.volumes:
            volume.neighbors = [self.faces[volume.faces[i]].volumes[int(not volume.flipped[i])] if not self.faces[volume.faces[i]].boundary else -1 for i in range(len(volume.faces))]
        self.pcount = len(self.points)
        self.fcount = len(self.faces)
        self.vcount = len(self.volumes)
        print(f"=== MESH CREATED ===")
        print(f" - {self.pcount} points")
        print(f" - {self.fcount} faces")
        print(f" - {self.vcount} volumes")
            