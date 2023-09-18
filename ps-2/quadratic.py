import numpy as np


def quadratic(a,b,c):
    delta = np.sqrt(b**2 - 4*a*c)
    if(b>=0):
        x2 = ( -b - delta ) / (2*a)
        x1 = (2*c) / ( -b - delta )
    else:
        x1 = ( -b + delta ) / (2*a)
        x2 = (2*c) / ( -b + delta )
    return x1,x2