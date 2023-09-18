import numpy as np

# Part a

def standard_solver(a,b,c):
    
    delta = np.sqrt(b**2 - 4*a*c)
    x1 = ( -b + delta ) / (2*a)
    x2= ( -b - delta ) / (2*a)
    return x1,x2


print(standard_solver(0.001,1000,0.001))

# Part b


def modified_solver(a,b,c):
    # this version multiplies ( -b +- np.sqrt(b**2 - 4*a*c) ) to top and bottom
    delta = np.sqrt(b**2 - 4*a*c)
    x1 = (2*c) / ( -b - delta )
    x2 = (2*c) / ( -b + delta )
    return x1,x2

print(modified_solver(0.001,1000,0.001))

# Notice that there's slight difference between the two results we got.

# The reason of the difference is that the computer only represents numbers to
# 16 significant figures, which is just 10^16 digits here. In this process, the
# approximation in the numbers caused the error.