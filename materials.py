import numpy as np

def umat (id,deps,sig,cm):
    if id == 1:
        return umat_001 (deps,sig,cm)
    else:
        raise Exception("ERROR: Mat ID not found")

#elastic material (supa simple)
def umat_001 (deps,sig,cm):
    #cm[1] = E (Young's Modulus)
    #cm[2] = v (Poisson's Ratio)
    
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