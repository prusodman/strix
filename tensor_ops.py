import numpy as np
#
#  an mxn matrix [m][n]
#  
#  m rows V // n columns >
#

def symm_sq (a):
    at = np.transpose(a)
    return 0.5*a+0.5*at

def skew_sq (a):
    at = np.transpose(a)
    return 0.5*a-0.5*at

def second_to_voigt (a):
    return [a[0][0],a[1][1],a[2][2],a[1][2],a[0][2],a[0][1]]
    
def voigt_to_second (a):
    return [[a[0],a[5],a[4]],[a[5],a[1],a[3]],[a[4],a[3],a[2]]]


def convert(val):
    possible = [str, int, float]
    val = val.rstrip()
    for func in possible:
        try:
            result = func(val)
        except ValueError:
            continue
        if str(result) == val and type(result) is not str:
            return result
    return val