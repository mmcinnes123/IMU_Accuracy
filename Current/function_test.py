import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root

# wM = np.array([2.84432866,0.57534497,-1.21490446])
# wI = np.array([0.24189251,5.47979496,0.72829094])
# q_exp =[-0.10942354,  0.72747063,  0.61280301, -0.28859219]

wM = np.array([0.25167923, -0.13302065,  0.95862569])
wI = np.array([-0.72116509, -0.39323903,  0.57033658])


wM_x = wM[0]
wM_y = wM[1]
wM_z = wM[2]
wI_x = wI[0]
wI_y = wI[1]
wI_z = wI[2]

weight = 10.0



def f(wxyz):
    q0 = wxyz[0]
    q1 = wxyz[1]
    q2 = wxyz[2]
    q3 = wxyz[3]

    f1 = (wM_x - wI_x) * q0 - (wM_z + wI_z) * q2 + (wM_y + wI_y) * q3
    f2 = (wM_y - wI_y) * q0 + (wM_z + wI_z) * q1 - (wM_x + wI_x) * q3
    f3 = (wM_z - wI_z) * q0 - (wM_y + wI_y) * q1 + (wM_x + wI_x) * q2
    f4 = weight * (q0**2 + q1**2 + q2**2 + q3**2 - 1)

    return np.array([f1, f2, f3, f4])

wxyz_0 = np.array([0, 1.0,0,0])
options_dict = {'maxiter':1000}
wxyz = root(f, wxyz_0, method='lm')
# q0 = wxyz[0]
# q1 = wxyz[1]
# q2 = wxyz[2]
# q3 = wxyz[3]
print(wxyz)
print(f(wxyz.x))


# wxyz_norm = wxyz / np.linalg.norm(wxyz)

