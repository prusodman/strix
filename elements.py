import numpy as np
import copy
import tensor_ops as tops
from materials import *

#PARENT ELEMENT CLASS
# THIS IS A PARENT CLASS TO ALLOW THE A PROBLEM TO 
# CONTAIN MULTIPLE TYPES OF ELEMENTS
#
#
# an element is a "live" object
# the solver will keep the values in the element up to date
# the element acts on the data contained within itself
#
#
class element:
    def __init__(self,data):
        self.eid = data[0]
        self.mid = data[1]
        self.con = data[2:]

        
#               HEX 8 ELEMENT
#               #############
#
#                7---------8
#               /|        /|
#              / |       / |
#             /  |      /  |
#            5---4-----6---3   
#            |  /      |  /
#            | /       | /
#            |/        |/
#            1---------2
#
#
#SINGLE INTEGRATION POINT HEX ELEMENT
class hex8 (element):
    def __init__(self,data):
        #call parent initializer
        super().__init__(data)
        #define space to place nodal data
        self.n0 = [] # original nodal coordinates
        self.n1 = [] # current nodal coordinates
        #define space to place deformation gradients
        self.F0 = np.identity(3)
        self.F1 = np.identity(3)
        #define space to place strain and strain rate
        self.ep = np.zeros((3,3)) #strain
        self.deps = np.zeros((3,3)) #strain rate
        self.sig = np.zeros((3,3))
        #define space to place history vars
        self.hisv = []*8
    
    #get nodal displacements (for all 8 nodes in element)
    def get_nodal_displacement (self):
        #convert node lists into arrays to perform math
        pos0 = np.array(self.n0)
        pos1 = np.array(self.n1)
        #return displacements
        return pos1 - pos0
    
    #convert position list to array for math reasons
    def get_nodal_positions (self):
        return np.array(self.n1)
    
    #get displacement within element using shape function interpolation
    def get_displacement (self,r,s,t):
        #nodal displacement
        nodes = np.array(self.get_nodal_displacement())
        #get shape function
        N = self.get_N (r,s,t)
        #interpolate
        u = N @ nodes
        return u.tolist()
    
    #from here: https://www.osti.gov/servlets/purl/632793
    #Efficient Computation of Volume of Hexahedral Cells
    def get_volume (self):
        d70 = np.array(self.n1[6]) - np.array(self.n1[0])
        d10 = np.array(self.n1[1]) - np.array(self.n1[0])
        d40 = np.array(self.n1[4]) - np.array(self.n1[0])
        d20 = np.array(self.n1[3]) - np.array(self.n1[0])
        d35 = np.array(self.n1[2]) - np.array(self.n1[5])
        d56 = np.array(self.n1[5]) - np.array(self.n1[7])
        d63 = np.array(self.n1[7]) - np.array(self.n1[2])
     
        v1 = np.linalg.det(np.array([d70,d10,d35]))
        v2 = np.linalg.det(np.array([d70,d40,d56]))
        v3 = np.linalg.det(np.array([d70,d20,d63]))
        v = (v1+v2+v3)/6.0
        return v
        
    def get_mass (self,mat):
        density =float(mat.density)
        volume = float(self.get_volume())
        print(density)
        print(volume)
        return volume*density
    
    def get_nmass (self,mat):
        return self.get_mass(mat)/8.0
    
    #get shape function wrt natural coordinates (chat GPT)
    def get_N (self,r,s,t):
        N = np.array([(0.125)*(1-r)*(1-s)*(1-t),
                     (0.125)*(1+r)*(1-s)*(1-t),
                     (0.125)*(1+r)*(1+s)*(1-t),
                     (0.125)*(1-r)*(1+s)*(1-t),
                     (0.125)*(1-r)*(1-s)*(1+t),
                     (0.125)*(1+r)*(1-s)*(1+t),
                     (0.125)*(1+r)*(1+s)*(1+t),
                     (0.125)*(1-r)*(1+s)*(1+t)])
        return N
    
    #shape function derivatives with respect to natural coordinates, (chat GPT)    
    def get_dN (self,r,s,t):
        dN_dr = np.array([-0.125*(1-s)*(1-t),
                           0.125*(1-s)*(1-t),
                           0.125*(1+s)*(1-t),
                          -0.125*(1+s)*(1-t),
                          -0.125*(1-s)*(1+t),
                           0.125*(1-s)*(1+t), 
                           0.125*(1+s)*(1+t),
                          -0.125*(1+s)*(1+t)])
        
        dN_ds = np.array([-0.125*(1-r)*(1-t),
                          -0.125*(1+r)*(1-t),
                           0.125*(1+r)*(1-t),
                           0.125*(1-r)*(1-t),
                          -0.125*(1-r)*(1+t),
                          -0.125*(1+r)*(1+t),
                           0.125*(1+r)*(1+t),
                           0.125*(1-r)*(1+t)])
        
        dN_dt = np.array([-0.125*(1-r)*(1-s),
                          -0.125*(1+r)*(1-s),
                          -0.125*(1+r)*(1+s),
                          -0.125*(1-r)*(1+s),
                           0.125*(1-r)*(1-s),
                           0.125*(1+r)*(1-s),
                           0.125*(1+r)*(1+s),
                           0.125*(1-r)*(1+s)])
        
        return [dN_dr,dN_ds,dN_dt]

    #Jacobian matrix
    def get_J (self,r,s,t):
        [dN_dr,dN_ds,dN_dt] = self.get_dN (r,s,t)
        scl = np.array([[-1, 1, 1, -1, -1, 1, 1, -1],
                        [-1, -1, 1, 1, -1, -1, 1, 1],
                        [-1, -1, -1, -1, 1, 1, 1, 1]])
        J = np.array([dN_dr, dN_ds, dN_dt]).dot(np.transpose(scl))
        return J
    
    #B matrix (strain displacement), (chat GPT)
    #B = LN = 
    def get_B (self,r,s,t):
        [dN_dr,dN_ds,dN_dt] = self.get_dN (r,s,t)
        B = np.zeros((6, 24))
        for i in range(8):
            B[0, i*3] = dN_dr[i]
            B[1, i*3 + 1] = dN_ds[i]
            B[2, i*3 + 2] = dN_dt[i]
            B[3, i*3] = dN_ds[i]
            B[3, i*3 + 1] = dN_dr[i]
            B[4, i*3 + 1] = dN_dt[i]
            B[4, i*3 + 2] = dN_ds[i]
            B[5, i*3] = dN_dt[i]
            B[5, i*3 + 2] = dN_dr[i]
        return B
    
    #get deformation gradient using method described here:
    #https://www.continuummechanics.org/finiteelementmapping.html
    #TODO: can I simplify this with the N / dN matrices
    def get_def_grad (self,r,s,t):
        #get nodal positions and displacement
        U = self.get_nodal_displacement()
        P = self.get_nodal_positions()
        
        #initialize matrix of derivatives
        dispDbasis = np.zeros((3,3)) #(du/dr)
        posiDbasis = np.zeros((3,3)) #(dX/dr)
        
        #determine sign for term
        #r toggles every two elements (- ++ -- ++ -)
        #s toggles every two elements (-- ++ -- ++)
        #t toggles every four elements (---- ++++)
        m = [[-1,1,1,-1,-1,1,1,-1],
             [-1,-1,1,1,-1,-1,1,1],
             [-1,-1,-1,-1,1,1,1,1]]
        
        #i = displacement/position (u,v,w or X,Y,Z)
        #j = basis function (r,s,t)
        #node = node number
        for i in range(0,3):
            for j in range (0,3):
                for node in range (0,8):
                    #DEBUGGING output    
                    #print(i,j,node,m[j][node])
                    #assemble terms in each derivative
                    dispDbasis [i][j] += m[j][node]*U[node][i]
                    posiDbasis [i][j] += m[j][node]*P[node][i]
        #return def. grad
        # F = I + [du/dr]*inv([dX/dr])
        return np.identity(3) +  dispDbasis @ np.linalg.inv(posiDbasis)
    
    #TODO: I need to check this!! This was generated by chatgpt. It makes some 
    #sense to me, it uses the strain displacement matrix, jacobian and the stresses
    # map the stresses to the 8 nodes BUT I don't understand the full underlying math...
    def get_force (self,r,s,t):
        ## my super basic understanding here
        ## stress*strain = volumetric energy / displacement = Force (Work = F*s)
        ## super dumbed down, there are tensor expressions for these
        BT = np.transpose(self.get_B (r,s,t))
        detJ = np.linalg.det(self.get_J (r,s,t))
        S = np.array(tops.second_to_voigt(self.sig))
        f = np.dot(BT,S)*detJ
        return f.reshape(8,3)
        
    
    #update element stresses and strains based on deformation gradient
    def update (self,b,mat):
        
        self.F0 = copy.deepcopy(self.F1)
        self.F1 = self.get_def_grad(0,0,0)

        dF = self.F1-self.F0
        #b = 0 backwards
        #b = 0.5 midpoint
        #b = 1 forward
        
        Fb = self.F0-self.F0*b+self.F1*b
        Finv = np.linalg.inv(Fb)
        Lb = dF @ Finv
        Db = tops.symm_sq(Lb)
        Wb = tops.skew_sq(Lb)
        R = Wb
        
        #TODO: not sure if objective update is working
        # .    find a way to check
        #objective rate (Jaumann rate = W)
        self.deps = Db - self.ep @ R + R @ self.ep
        self.ep = self.ep + self.deps
        
        deps_v = tops.second_to_voigt(self.deps)
        sig0_v = tops.second_to_voigt(self.sig)
        sig1_v = umat(mat.mat,deps_v,sig0_v,mat.cm)
        
        #OBJECTIVE UPDATE OF STRESS
        self.sig = tops.voigt_to_second(sig1_v)
        self.sig = self.sig - self.sig @ R + R * self.sig
        