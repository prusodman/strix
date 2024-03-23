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
        self.con = data[1:]

        
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
        nodes = self.get_nodal_displacement()
        u1 = np.array(nodes[0])
        u2 = np.array(nodes[1])
        u3 = np.array(nodes[2])
        u4 = np.array(nodes[3])
        u5 = np.array(nodes[4])
        u6 = np.array(nodes[5])
        u7 = np.array(nodes[6])
        u8 = np.array(nodes[7])
        # interpolate nodal displacements using shape function
        # https://www.continuummechanics.org/finiteelementmapping.html
        u =  0.125*(1-r)*(1-s)*(1-t)*u1\
            +0.125*(1+r)*(1-s)*(1-t)*u2\
            +0.125*(1+r)*(1+s)*(1-t)*u3\
            +0.125*(1-r)*(1+s)*(1-t)*u4\
            +0.125*(1-r)*(1-s)*(1+t)*u5\
            +0.125*(1+r)*(1-s)*(1+t)*u6\
            +0.125*(1+r)*(1+s)*(1+t)*u7\
            +0.125*(1-r)*(1+s)*(1+t)*u8
        return u.tolist()
        
    #get deformation gradient using method described here:
    #https://www.continuummechanics.org/finiteelementmapping.html
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
    #update element stresses and strains based on deformation gradient
    def update (self,b):
        
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
        sig1_v = umat(1,deps_v,sig0_v,[207000,0.3])
        
        #OBJECTIVE UPDATE OF STRESS
        self.sig = tops.voigt_to_second(sig1_v)
        self.sig = self.sig - self.sig @ R + R * self.sig
        