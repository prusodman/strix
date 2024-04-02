#  ___ _____ ___ _____  __   __   __  
# / __|_   _| _ \_ _\ \/ / </  \^/  \>
# \__ \ | | |   /| | >  <  | () | () |
# |___/ |_| |_|_\___/_/\_\  \__\_/__/ 
#
# STRIX SOLVER  v0.2
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
# - understand and develop explicit loop            [ ]
#   > determine internal forces in element          [ ]
#   > compute external forces                       [ ]
# - get strain and strain rate to feed into UMATS   [X]
# - develop elastic UMAT                            [X]
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
        self.n0 = []
        self.n1 = []
        self.elements = []
        self.bcs = []
        self.F = []
        self.materials = []
        self.mass = []
    
    def read_file (self,fname):
        deck = open (fname,'r')
        lines = deck.readlines()
        #input type
        # 0 = none
        # 1 = node
        # 2 = element
        # 3 = BC
        # 4 = material
        #99 = title
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
                    self.elements.append(hex8(cnvt))
                elif itype == 3:
                    self.bcs.append(cnvt)
                elif itype == 4:
                    self.materials.append(material(cnvt[0],cnvt[1],cnvt[2],cnvt[3:]))
                elif itype == 99:
                    self.title = line.rstrip()
        
            
    ### STRIX SOLVERS ###
    # strix explicit solver < I'm not making any more XD, explicit is da best
    def strix_explicit (self,dt,Tf):
        print (self.header())
        print("\n",self.title)
        
        tic = time.time()
        #get time step given total simulation time and timestep
        ts = Tf/dt
        #initialize increment counter
        inc = 0
        
        #prime past and present node values
        self.n1 = copy.deepcopy(self.n0)
        
        #initialize positions in element
        for ele in self.elements:
            self.nodal_init(ele.eid)
        
        self.mass_init()
        
        #solver loop
        while inc < ts:
            #BC UPDATE
            for bc in self.bcs:
                #parse bcs
                nid = bc[0] #nodal id
                dof = bc[1] #degree of freedom to apply BC
                mag = bc[2] #magnitude of BC (postion, velocity ...)
                typ = bc[3] #type of BC
                
                #get node key from node id (these are separate)
                key = self.get_nodekey(nid)
                
                #apply boundary condition based on type
                if typ == 1: #position BC
                   self.n1[key][dof] = self.n0[key][dof]+mag
                elif typ == 2: #velocity BC
                   self.n1[key][dof] = self.n1[key][dof] + mag*dt
                #elif typ == 4: #force BC (not implemented yet)
                    #TODO: code force BC 
                else:
                    print (typ)
                    raise Exception("ERROR: BC type not found")
            
            #ELEMENT UPDATE
            for ele in self.elements:
                #get current positions of nodes
                self.nodal_update(ele.eid)
                matid = self.get_materialkey (ele.mid)
                mat = self.materials[matid]
                # do not run element update until all variables are primed
                if inc > 0:
                   ele.update(0.5,mat)
            
            #FORCE UPDATE
            size = len(self.n1)
            GF = np.zeros((size,3))
            for ele in self.elements:
                F = ele.get_force(0,0,0)
                cntr = 0
                for con in ele.con:
                    key = self.get_nodekey (con)
                    GF[key] += F[cntr]
                    
            
                
            #INCREMENT counter, print every 100 cycles
            if inc%100 == 0:
                #keep track of seconds elapsed
                toc = time.time()
                #print cycles and elapsed time
                print ("Cycle ",inc,": ",round(toc-tic,1),"s elapsed")
            #INCREMENT time
            inc += 1
        print ("Finished solving in ",round(toc-tic,1),"s")
            
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
    
    #initiates initial nodal positions of an element
    def nodal_init (self,eid):
        #get element key
        ekey = self.get_elementkey (eid)
        #get list of node keys in element
        nlist = self.get_nkey_in_element(eid)
        #clear node list in element
        self.elements[ekey].n0 = []
        #iterate through nodes in element
        for nid in nlist:
            #add nodes to element
            self.elements[ekey].n0.append(self.n0[nid][1:])
        #copy initial nodal positions to current nodal positions
        self.elements[ekey].n1 = copy.deepcopy(self.elements[ekey].n0)
    
    #fill in mass matrix
    def mass_init (self):
        self.mass = np.zeros((len(self.n1)))
        for ele in self.elements:
            mat = self.materials[self.get_materialkey(ele.mid)]
            nmass = ele.get_nmass(mat)
            for con in ele.con:
                nid = self.get_nodekey(con)
                self.mass[nid] = self.mass[nid] + nmass
    
    #update current nodal positions of an element
    def nodal_update (self,eid):
        #get element key
        ekey = self.get_elementkey (eid)
        #get lit of node keys in element
        nlist = self.get_nkey_in_element(eid)
        #iterate through nodes in element and update positions
        cntr = 0;
        for nid in nlist:
            self.elements[ekey].n1[cntr] = self.n1[nid][1:]
            cntr = cntr + 1;

    def header (self):
        return '   ___ _____ ___ _____  __   __   __   '+'\n'+\
               '  / __|_   _| _ \_ _\ \/ / </  \^/  \> '+'\n'+\
               '  \__ \ | | |   /| | >  <  | () | () | '+'\n'+\
               '  |___/ |_| |_|_\___/_/\_\  \__\_/__/  '+'\n'+\
               '  ==================================== '+'\n'+\
               '  Solver Version - 0.2                 '+'\n'