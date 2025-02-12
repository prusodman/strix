import numpy as np
import math
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
        
        #define space to place nodal data
        #define space to place deformation gradients
        self.F0 = np.identity(3)
        self.F1 = np.identity(3)
        #define space to place strain and strain rate
        self.ep = np.zeros((3,3)) #strain
        self.deps = np.zeros((3,3)) #strain rate
        self.sig = np.zeros((3,3))
        #define space to place history vars
        self.hisv = []*8
        self.clen = 0
    
    #update element stresses and strains based on deformation gradient
    def update (self,b,mat,P,U):
        self.F0 = copy.deepcopy(self.F1)
        self.F1 = self.get_def_grad(P,U)

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
        sig1_v = umat(mat.mat,deps_v,sig0_v,mat.cm,self.hisv)
        
        #TODO: need to update characteristic length
        c = mat.get_c();
        clen = 1;
        
        #BULK VISCOSITY IMPLEMENTATION
        Q1 = 1.50
        Q2 = 0.06
        
        #TODO: I am currently applying this to the strain rate tensor
        # I think this needs to be a scalar and added as a pressure
        # I think, I really don't understand bulk viscosity implementation
        effd = tops.eff_strain_v (deps_v)
        q = mat.density*clen*(Q1*clen*effd**2-Q2*c*effd)
        q_v = [q,q,q,0,0,0]
        
        #OBJECTIVE UPDATE OF STRESS
        self.sig = tops.voigt_to_second(sig1_v+q_v)
        self.sig = self.sig - self.sig @ R + R * self.sig
        
        

        
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
        self.nnum = 8
    
    #from here: https://www.osti.gov/servlets/purl/632793
    #Efficient Computation of Volume of Hexahedral Cells
    def get_volume (self,P):
        d70 = np.array(P[6]) - np.array(P[0])
        d10 = np.array(P[1]) - np.array(P[0])
        d40 = np.array(P[4]) - np.array(P[0])
        d20 = np.array(P[3]) - np.array(P[0])
        d35 = np.array(P[2]) - np.array(P[5])
        d56 = np.array(P[5]) - np.array(P[7])
        d63 = np.array(P[7]) - np.array(P[2])
     
        v1 = np.linalg.det(np.array([d70,d10,d35]))
        v2 = np.linalg.det(np.array([d70,d40,d56]))
        v3 = np.linalg.det(np.array([d70,d20,d63]))
        v = (v1+v2+v3)/6.0
        return v
        
    def get_mass (self,mat,P):
        density = float(mat.density)
        volume = float(self.get_volume(P))
        return volume*density
    
    def get_nmass (self,mat,P):
        return self.get_mass(mat,P)/8.0
    
    #lc = volume / aemax
    def get_dt (self,mat,P):
        V = self.get_volume(P)
        
        area = np.zeros(6)
        area [0] = tops.get_face_area (P[0],P[1],P[4])
        area [1] = tops.get_face_area (P[1],P[5],P[2])
        area [2] = tops.get_face_area (P[2],P[7],P[3])
        area [3] = tops.get_face_area (P[3],P[6],P[0])
        area [4] = tops.get_face_area (P[4],P[5],P[6])
        area [5] = tops.get_face_area (P[5],P[1],P[3])
        
        mina = min (area)
        lc = V/mina
        c = mat.get_c ()
        return lc/c
        
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
    #I NEED TO CALCULATE THIS
    def get_def_grad (self,P,U):
        # def get_def_grad (self,U,P):
        #get nodal positions and displacement
        
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
    
    #is detJ actually the volume (I DON'T THINK IT IS)
    def get_force (self,U,P):
        #weight of each gauss point for single integration point element
        gauss_weight = 1.0
        ## my super basic understanding here
        ## stress*strain = volumetric energy / displacement = Force (Work = F*s)
        ## super dumbed down, there are tensor expressions for these
        BT = np.transpose(self.get_B (0,0,0))
        detJ = np.linalg.det(self.get_J (0,0,0))
        S = np.array(tops.second_to_voigt(self.sig))
        f = gauss_weight*np.dot(BT,S)*detJ
        return f.reshape(8,3)
#Z
#
#               TET 4 ELEMENT
#               =============
#
#
#                     4
#                    /|\
#                   / | \
#                  /  |  \
#                 /   |   \
#               3/....|....\2
#                \    |    /
#                 \   |   /
#                  \  |  /
#                   \ | /
#                    \|/
#                     1
#
#
#
#SINGLE INTEGRATION POINT TET ELEMENT
# UNTESTED!!
class tet4 (element):
    def __init__(self,data):
        #call parent initializer
        super().__init__(data)
        self.nnum = 4
    
    def get_volume (self,P):
        #return np.abs(np.dot(P[0] - P[3], np.cross(P[1] - P[3], P[2] - P[3]))) / 6.0
        B1 = np.subtract(P[0],P[3])
        B2 = np.subtract(P[1],P[3])
        B3 = np.subtract(P[2],P[3])
        #print (P)
        return np.abs(np.dot(B1,np.cross(B2,B3)))/6.0
    
    def get_mass (self,mat,P):
        density = float(mat.density)
        volume = float(self.get_volume(P))
        return volume*density
    
    def get_nmass (self,mat,P):
        return self.get_mass(mat,P)/4.0
    
    def get_dt (self,mat,P):
        V = self.get_volume(self,P)
    
    def get_dN (self):
         # Derivatives of the shape functions with respect to local coordinates
        return np.array([
            [-1, -1, -1],
            [ 1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  1]
        ])
    
    def get_force (self, P,U):
        V = self.get_volume (P)
        B = self.get_dN ()
        nodal_forces = np.zeros((4, 3))  # 4 nodes, 3 force components per node
        for i in range(4):
            nodal_forces[i, :] = V * np.dot(B[i, :], self.sig)
        return nodal_forces
    
    def get_def_grad (self,P,U):
        #get nodal positions and displacement
        
        #initialize matrix of derivatives
        dispDbasis = np.zeros((3,3)) #(du/dr)
        posiDbasis = np.zeros((3,3)) #(dX/dr)
        
        #determine sign for term
        m = [[1,0,0,-1],
             [0,1,0,-1],
             [0,0,1,-1]]
        
        #i = displacement/position (u,v,w or X,Y,Z)
        #j = basis function (r,s,t)
        #node = node number
        for i in range(0,3):
            for j in range (0,3):
                for node in range (0,4):
                    #DEBUGGING output    
                    #print(i,j,node,m[j][node])
                    #assemble terms in each derivative
                    dispDbasis [i][j] += m[j][node]*U[node][i]
                    posiDbasis [i][j] += m[j][node]*P[node][i]
        #return def. grad
        # F = I + [du/dr]*inv([dX/dr])
        return np.identity(3) +  dispDbasis @ np.linalg.inv(posiDbasis)

    
    #lc = min altitude
    def get_dt (self,mat,P):
        
        V = self.get_volume(P)
        
        lc = V/mina
        c = mat.get_c ()
        return lc/c