
from importlib import reload
import pyvista as pv
from lapy import Solver, Plot, TriaMesh
reload(Plot)
import scipy.io as sio
import plotly.io as pio
pio.renderers.default='browser'
mesh = pv.read('InputMesh.stl')
points = mesh.points
points[:,2] = points[:,2]

faces = mesh.faces.reshape(-1, 4)
faces = faces[:,1:4]

k = 128
tetra = TriaMesh(points,faces)
fem   = Solver(tetra)
evals, evecs = fem.eigs(k = k)
evDict = dict()
evDict['Refine'] = 0
evDict['Degree'] = 1
evDict['Dimension'] = 2
evDict['Elements'] = len(tetra.t)
evDict['DoF'] = len(tetra.v)
evDict['NumEW'] = k
evDict['Eigenvalues']  = evals
evDict['Eigenvectors'] = evecs
evDict['Points'] = points
evDict['Faces'] = faces

sio.savemat('lbe_ev_input.mat', evDict)  
