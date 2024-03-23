#  ___ _____ ___ _____  __   __   __  
# / __|_   _| _ \_ _\ \/ / </  \^/  \>
# \__ \ | | |   /| | >  <  | () | () |
# |___/ |_| |_|_\___/_/\_\  \__\_/__/ 
#
# STRIX SOLVER  v0.1
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
        self.n0 = []
        self.n1 = []
        self.elements = []
        self.bcs = []
            
    ### STRIX SOLVERS ###
    # strix explicit solver < I'm not making any more XD, explicit is da best
    def strix_explicit (self,dt,Tf,title):
        print (self.header())
        print("\n",title)
        
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
        
        #solver loop
        while inc < ts:
            #BC HANDLING
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
                    raise Exception("ERROR: BC type not found")
            
            #ELEMENT UPDATE
            for ele in self.elements:
                #get current positions of nodes
                self.nodal_update(ele.eid)
                # do not run element update until all variables are primed
                if inc > 0:
                   ele.update(0.5)
                
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
    
    #update current nodal positions of an element
    def nodal_update (self,eid):
        #get element key
        ekey = self.get_elementkey (eid)
        #get lit of node keys in element
        nlist = self.get_nkey_in_element(eid)
        #clear node list in element
        self.elements[ekey].n1 = []
        #iterate through nodes in element and update positions
        for nid in nlist:
            self.elements[ekey].n1.append(self.n1[nid][1:])
    
    def header (self):
        return '   ___ _____ ___ _____  __   __   __   '+'\n'+\
               '  / __|_   _| _ \_ _\ \/ / </  \^/  \> '+'\n'+\
               '  \__ \ | | |   /| | >  <  | () | () | '+'\n'+\
               '  |___/ |_| |_|_\___/_/\_\  \__\_/__/  '+'\n'
               