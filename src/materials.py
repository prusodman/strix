import numpy as np
import math

# create container class to hold material data
class material:
    # mid = material id
    # mat = material card
    # cm = vars
    def __init__(self,mid,mat,density,cm):
        self.mid = mid
        self.mat = mat
        self.density = float(density)
        self.cm = cm
        
    def get_c (self):
        #cm[0] = E (Young's Modulus)
        #cm[1] = v (Poisson's Ratio)
        return math.sqrt ((self.cm[0]*(1-self.cm[1]))/((1+self.cm[1])*(1-2*self.cm[1])*self.density))

#class to find and call materials
def umat (mid,deps,sig,cm,hisv):
    if mid == 1:
        return umat_001 (deps,sig,cm)
    else:
        raise Exception("ERROR: Mat ID not found")

#elastic material (supa simple)
def umat_001 (deps,sig,cm):
    #cm[0] = E (Young's Modulus)
    #cm[1] = v (Poisson's Ratio)
    
    L4 = np.array(get_L4 (cm[0],cm[1]))
    #print(L4.tolist())
    dsig = L4 @ np.array(deps)
    return (sig + dsig).tolist()

def get_L4 (E,v):
    lm = E*v/((1+v)*(1-2*v))
    #mu is G (shear modulus)
    mu = E/(2+2*v)
    c1 = 2*mu+lm
    return  [[c1,lm,lm, 0, 0, 0],\
            [lm,c1,lm, 0, 0, 0],\
            [lm,lm,c1, 0, 0, 0],\
            [ 0, 0, 0, mu,0, 0],\
            [ 0, 0, 0, 0, mu,0],\
            [ 0, 0, 0, 0, 0,mu]]