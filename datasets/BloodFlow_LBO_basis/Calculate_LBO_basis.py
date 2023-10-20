
from importlib import reload
from lapy import Solver, Plot, TetMesh
reload(Plot)
import scipy.io as sio
import plotly.io as pio
pio.renderers.default='browser'
import pandas as pd

points = pd.read_csv('coordinates.csv',header=None).values
faces = pd.read_csv('elements.csv',header=None).values

points = points
faces = faces-1

k = 128
tetra = TetMesh(points,faces)
fem = Solver(tetra)
evals, evecs = fem.eigs(k=k)


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

sio.savemat('LBO_basis.mat', evDict)  


