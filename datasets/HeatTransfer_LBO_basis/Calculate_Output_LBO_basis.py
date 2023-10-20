
from importlib import reload
from lapy import Solver, Plot, TetMesh
reload(Plot)
import scipy.io as sio
import plotly.io as pio
pio.renderers.default='browser'
mesh   = sio.loadmat('Part_Mesh.mat')
points = mesh['points']
faces  = mesh['elements'] - 1
faces  = faces.astype(int)

k = 128
tetra = TetMesh(points,faces)
fem   = Solver(tetra)
evals, evecs = fem.eigs(k = k)
evDict = dict()
evDict['Refine'] = 0
evDict['Degree'] = 1
evDict['Dimension'] = 2
evDict['Elements'] = len(tetra.t)
evDict['DoF'] = len(tetra.v)
evDict['NumEW'] = k
evDict['Eigenvalues'] = evals
evDict['Eigenvectors'] = evecs
evDict['Points'] = points
evDict['Faces'] = faces

print(evecs.shape)

sio.savemat('lbe_ev_output.mat', evDict)  
