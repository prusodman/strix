#  ___ _____ ___ _____  __   __   __  
# / __|_   _| _ \_ _\ \/ / </  \^/  \>
# \__ \ | | |   /| | >  <  | () | () |
# |___/ |_| |_|_\___/_/\_\  \__\_/__/ 
#
# STRIX SOLVER  v0.4
# ==================
# Developed by Prusodman Sathananthan
# March 17th, 2024 (St. Paddy's Day lol)
#
# GOAL: Educational explicit dynamic finite element
# solver to use to better understand FEM + develop
# novel user materials quickly and without cost
#
# CHANGELOG
# =========
# v0.1 - 20240317 __ Initial Creation
# v0.2 - 20240323 __ Completed Single Element Logic
# v0.3 - 20240523 __ Added input deck functionality + file output
# v0.4 - 20240524 __ Redid explicit loop (MASSIVE CHANGES)
#
# X - Done
# * - In Progress
# O - Ongoing Work
#
# TODO
# ====
# - get def. grad from displacements                [X]
# - create math library to support tensor math      [X]
# - make stress update method                       [X]
# - understand and develop explicit loop            [X]
#   > determine internal forces in element          [X]
#   > compute external forces                       [ ]
# - get strain and strain rate to feed into UMATS   [X]
# - develop elastic UMAT                            [X]
# - refactor and clean up                           [O]
# - comment what you have so far...                 [O]
#
#
#
import numpy as np
import time
import copy
import tensor_ops as tops
from elements import *
#
# THE SOLVER CLASS HAS THE FOLLOWING JOBS
# 1. Holds all the data for simulation
# 2. Updates elements
# 3. Performs analysis loops
#

class strix():
    # Intialization of solver
    # Variables
    # n0 = past nodal values
    # n1 = present nodal values
    # elements = list of elements in problem
    # bcs = list of bcs in problem
    def __init__(self):
        self.title = ""
        
        self.n0 = [] #position
        self.u = [] #displacement
        self.v = [] #velocity
        self.a = [] #acceleration
        self.mass = []
        
        self.Fint = []
        self.Fext = []
        
        self.elements = []
        self.bcs = []
        self.materials = []
        
        self.dt = 0.0
        self.Tf = 0.0
        self.T = 0.0
        self.fout = 0.0
        self.fdir = ""
    
    def read_file (self,fname):
        deck = open (fname,'r')
        lines = deck.readlines()
        #input type
        # 0 = none
        # 1 = node
        # 2 = element
        # 3 = BC
        # 4 = material
        # 1x = controls
        # 2x = outputs
        # 99 = title
        itype = 0
        count = 0 
        for line in lines:
            lead = line[0]
            header = line[1:].rstrip()
            if lead == '$':
                pass
            elif lead == '*':
                if header == "NODE":
                    itype = 1
                elif header == "ELEMENT":
                    itype = 2
                elif header == "BOUNDARY_CONDITION":
                    itype = 3
                elif header == "MATERIAL":
                    itype = 4
                elif header == "CONTROL_TIME":
                    itype = 10
                elif header == "CONTROL_OUTPUT":
                    itype = 11
                elif header == "OUTPUT_ELEMENT":
                    itype = 20
                elif header == "TITLE":
                    itype = 99
                else:
                    itype = 0
            else:
                data = line.rstrip().split(',')
                cnvt = [tops.convert(flag) for flag in data]
                if itype == 1:
                    self.n0.append(cnvt)
                elif itype == 2:
                    nargs = len (cnvt)
                    #if 10 args, its a hex8
                    if nargs == 10:
                        self.elements.append(hex8(cnvt))
                elif itype == 3:
                    self.bcs.append(cnvt)
                elif itype == 4:
                    self.materials.append(material(cnvt[0],cnvt[1],cnvt[2],cnvt[3:]))
                elif itype == 10:
                    self.Tf = float(cnvt[0])
                elif itype == 11:
                    self.fout = cnvt[0]
                    self.fdir = cnvt[1]
                elif itype == 20:
                    self.sets.append(cnvt)
                elif itype == 99:
                    self.title = line.rstrip()
        
            
    ### STRIX SOLVERS ###
    # strix explicit solver < I'm not making any more XD, explicit is da best
    def strix_explicit (self,Tf):
        
        self.initialize()
        tic = time.time()
        
        #ts = Tf/self.dt
        #initialize increment counter
        inc = 0
        #solver loop
        while self.T < Tf:
    
            self.update_velocity()
            self.update_disp()
            self.enforce_bcs()
            self.update_elements(inc)
            
            self.update_internalF ()
            self.update_externalF ()
            self.update_acceleration()
            
            #INCREMENT counter, print every 100 cycles
            if inc%500 == 0:
                #keep track of seconds elapsed
                toc = time.time()
                #print cycles and elapsed time
                print ("Cycle ",inc,":\t",round(toc-tic,1),"s elapsed"," dt={:.2e}".format(self.dt),"T={:.2e}".format(self.T))
            
            #output data
            if inc%self.fout == 0:
                self.output_data(inc)
                
            #INCREMENT time
            inc += 1
            self.T += self.dt
            
        print ("Finished solving in ",round(toc-tic,1),"s")
        self.output_data(inc)
        
    def initialize(self):
        print (self.header())
        print("\n",self.title)
        
        f = open (self.fdir,'w')
        f.write (self.header()+'\n')
        f.write (self.title+'\n')
        f.close()
         
        #number of nodes (for matricies)
        nnum = len(self.n0)
        
        #prime kinematics
        self.u = np.zeros((nnum,3))
        self.v = np.zeros((nnum,3))
        self.a = np.zeros((nnum,3))
        
        #prime forces
        self.Fint = np.zeros((nnum,3))
        self.Fext = np.zeros((nnum,3))
        
        self.mass_init()

        self.dt = 0.1/self.materials[0].get_c()
    
    def update_velocity (self):
        #number of nodes (for matricies)
        nnum = len(self.n0)
        for ind in range (nnum):
            self.v[ind] = self.v[ind] + self.dt*self.a[ind]
        
    def update_disp (self):
        #number of nodes (for matricies)
        nnum = len(self.n0)
        for ind in range (nnum):
            self.u[ind] = self.u[ind] + self.dt*self.v[ind]
            
    def enforce_bcs (self):
        for bc in self.bcs:
            #parse bcs
            nid = bc[0] #nodal id
            dof = bc[1] - 1 #degree of freedom to apply BC
            mag = bc[2] #magnitude of BC (postion, velocity ...)
            typ = bc[3] #type of BC
            
            #get node key from node id (these are separate)
            key = self.get_nodekey(nid)
            
            #apply boundary condition based on type
            if typ == 1: #position BC
                self.u[key][dof] = mag
                self.v[key][dof] = 0.0
                self.a[key][dof] = 0.0
            elif typ == 2: #velocity BC
                #undo velocity 
                self.u[key][dof] = self.u[key][dof] - self.v[key][dof]*self.dt
                self.v[key][dof] = mag
                self.a[key][dof] = 0.0
                #apply new velocity
                self.u[key][dof] = self.u[key][dof] + self.v[key][dof]*self.dt
            #elif typ == 4: #force BC (not implemented yet)
                #TODO: code force BC 
            else:
                print (typ)
                raise Exception("ERROR: BC type not found")
        
    def update_internalF (self):
        nnum = len(self.n0)
        self.Fint = np.zeros((nnum,3))
        for ele in self.elements:
            f = ele.get_force(0,0,0)
            cntr = 0
            for con in ele.con:
                nid = self.get_nodekey(con)
                self.Fint[nid][:] = self.Fint[nid][:] + f[cntr][:]
                cntr = cntr + 1
    
    def update_externalF (self):
        nnum = len(self.n0)
        self.Fext = np.zeros((nnum,3))
        
        for bc in self.bcs:
            #parse bcs
            nid = bc[0] #nodal id
            dof = bc[1]-1 #degree of freedom to apply BC
            mag = bc[2] #magnitude of BC (postion, velocity ...)
            typ = bc[3] #type of BC
            
            #get node key from node id (these are separate)
            key = self.get_nodekey(nid)
            
            #apply boundary condition based on type
            if typ == 1: #position BC
                self.Fext [key][dof] = -self.Fint [key][dof]
            elif typ == 2: #velocity BC
                self.Fext [key][dof] = -self.Fint [key][dof]
            #elif typ == 4: #force BC (not implemented yet)
                #TODO: code force BC 
            else:
                print (typ)
                raise Exception("ERROR: BC type not found")
                
            
    def update_acceleration (self):
        #number of nodes (for matricies)
        nnum = len(self.n0)
        for ind in range (nnum):
            self.a[ind] = (self.Fext[ind] - self.Fint[ind])/self.mass[ind]

    def update_elements (self,inc):
        for ele in self.elements:
            #get material of element
            matid = self.get_materialkey (ele.mid)
            mat = self.materials[matid]
            
            #get list of node keys in element
            nlist = self.get_nkey_in_element(ele.eid)
            
            #generate list of element initial positions and displacements
            cntr = 0;
            U = np.zeros((8,3))
            P0 = np.zeros((8,3))
            for nid in nlist:
                P0[cntr][:] = self.n0[nid][1:]
                U[cntr][:] = self.u[nid][:]
                cntr+=1
            
            # do not run element update until all variables are primed
            if inc > 0:
                ele.update(0.5,mat,U,U+P0)
    
            
    ### TRANSLATION FUNCTIONS
    ### Get list keys (python) from node/element ids (input deck)
    # get node key from node id
    def get_nodekey (self,nid):
        # make a list of node ids from element list
        list_id = [node[0] for node in self.n0]
        # find node id in list and return key
        return list_id.index(nid)
    
    # get element key from element id
    def get_elementkey (self,eid):
        # make a list of element ids from element list
        ele_id = [ele.eid for ele in self.elements]
        # find element id in list and return key
        return ele_id.index(eid)
    
    def get_materialkey (self,mid):
        # make a list of element ids from materials list
        mat_id = [mat.mid for mat in self.materials]
        # find material id in list and return key
        return mat_id.index(mid)
    
    # generates a list of node keys in an element
    def get_nkey_in_element (self,eid):
        # create an empty node key list
        nlist = []
        # get element from element list using element id
        ele = self.elements[self.get_elementkey(eid)]
        # iterate through element connectivity and get node keys
        for nid in ele.con:
            nlist.append(self.get_nodekey(nid))
        # return keys
        return nlist
    
    ### UPDATE FUNCTIONS
    ### updates nodal values of elements in current problem
    #TODO: Instead of clearing and readding nodes, try to update node values
    
    #fill in mass matrix
    def mass_init (self):
        nnum = len(self.n0)
        self.mass = np.zeros((nnum))
        for ele in self.elements:
            mat = self.materials[self.get_materialkey(ele.mid)]
            
            
            #get list of node keys in element
            nlist = self.get_nkey_in_element(ele.eid)
            
            #generate list of element initial positions and displacements
            cntr = 0;
            U = np.zeros((8,3))
            P0 = np.zeros((8,3))
            for nid in nlist:
                P0[cntr][:] = self.n0[nid][1:]
                U[cntr][:] = self.u[nid][:]
                cntr+=1
            
            P = U + P0
            nmass = ele.get_nmass(mat,P)
            for con in ele.con:
                nid = self.get_nodekey(con)
                self.mass[nid] = self.mass[nid] + nmass
    
    def output_data (self,inc):
        f = open (self.fdir,'a')
        f.write(">Cycle "+str(inc)+" - Time {:.2e}".format(self.T)+'\n')
        for ele in self.elements:
            sigout = tops.second_to_voigt(ele.sig)
            output = ["{:.5e}".format(x).rjust(14) for x in sigout]
            f.write('E'+str(ele.eid)+'\t')
            f.write('\t'.join(output))
            f.write('\n')
                
        nnum = [item[0] for item in self.n0]
        P0 = np.array(self.n0)
        P = np.add(P0[:,1:],self.u)
        
        #for node in P:
        #    output = ["{:.10e}".format(x) for x in node[1:]]
        #    f.write('N'+str(node[0])+'\t')
        #    f.write('\t'.join(output))
        #    f.write('\n')
        
        for i in range (len(self.n0)):
            output = ["{:.5e}".format(x).rjust(14) for x in P[i,:]]
            f.write('N'+str(nnum[i])+'\t')
            f.write('\t'.join(output))
            f.write('\n')
        
        for i in range (len(self.n0)):
            output = ["{:.5e}".format(x).rjust(14) for x in self.u[i,:]]
            f.write('U'+str(nnum[i])+'\t')
            f.write('\t'.join(output))
            f.write('\n')
        
        for i in range (len(self.n0)):
            output = ["{:.5e}".format(x).rjust(14) for x in self.Fext[i,:]]
            f.write('FE'+str(nnum[i])+'\t')
            f.write('\t'.join(output))
            f.write('\n')
        
        for i in range (len(self.n0)):
            output = ["{:.5e}".format(x).rjust(14) for x in self.Fint[i,:]]
            f.write('FI'+str(nnum[i])+'\t')
            f.write('\t'.join(output))
            f.write('\n')
        
        f.write("\n")
        f.close()
                
    def header (self):
        return '   ___ _____ ___ _____  __   __   __   '+'\n'+\
               '  / __|_   _| _ \_ _\ \/ / </  \^/  \> '+'\n'+\
               '  \__ \ | | |   /| | >  <  | () | () | '+'\n'+\
               '  |___/ |_| |_|_\___/_/\_\  \__\_/__/  '+'\n'+\
               '  ==================================== '+'\n'+\
               '  Solver Version - 0.4                 '+'\n'