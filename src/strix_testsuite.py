import sys
import strix as st
import tensor_ops as tops
from elements import *

def main():
    
    #UNIT TEST 1a: ELASTIC MAT
    print ("\nTesting MATERIALS...")
    print ("Creating elastic steel material")
    steel = material(100,1,7.8e-9,[207000,0.3])
    
    print ("Calculating wave speed...")
    long_ws = round(steel.get_c()/1000)
    print ("Wave speed: "+str(long_ws)+" m/s")
    if (long_ws == 5977):
        print ("--PASS--")
    else:
        print ("--FAIL--")
    print ("Done MATERIALS...\n")
    
    
    #UNIT TEST 2a: HEX ELEMENT
    print ("Testing HEX Element...")
    print ("Creating 10x10x10 element")
    
    n0 = [] #nodes
    P0 = [] #position
    U_UT = [] #displacement
    U_SS = [] #displacement

    n0.append([1,0,0,0])
    n0.append([2,10,0,0])
    n0.append([3,10,10,0])
    n0.append([4,0,10,0])
    n0.append([5,0,0,10])
    n0.append([6,10,0,10])
    n0.append([7,10,10,10])
    n0.append([8,0,10,10])
    
    U_UT.append([0,0,0])
    U_UT.append([-0.03,0,0])
    U_UT.append([-0.03,-0.03,0])
    U_UT.append([0,-0.03,0])
    U_UT.append([0,0,0.1])
    U_UT.append([-0.03,0,0.1])
    U_UT.append([-0.03,-0.03,0.1])
    U_UT.append([0,-0.03,0.1])

    U_SS.append([0,0,0])
    U_SS.append([0,0,0])
    U_SS.append([0,0,0])
    U_SS.append([0,0,0])
    U_SS.append([0,0.1,0])
    U_SS.append([0,0.1,0])
    U_SS.append([0,0.1,0])
    U_SS.append([0,0.1,0])
    
    
    #generate list of element positions
    for node in n0:
        P0.append (node[1:])
        
    hexa = hex8 ([1,100,1,2,3,4,5,6,7,8])
    print ("Calculating volume...")
    vol = round(hexa.get_volume(P0))
    print ("Volume: "+str(vol)+" mm^3")
    if (vol == 1000):
        print ("--PASS--")
    else:
        print ("--FAIL--")
        
    print ("Calculating mass...")    
    mass = round(hexa.get_mass (steel,P0)*1e6,9)
    print ("Mass: "+str(mass)+"g")
    if (mass == 7.8):
        print ("--PASS--")
    else:
        print ("--FAIL--")
        
    print ("Calculating timestep...")    
    dt = round(hexa.get_dt(steel,P0)*1e6,3)
    print ("dt: "+str(dt)+"us")
    if (dt == 1.673):
        print ("--PASS--")
    else:
        print ("--FAIL--")
    
    print ("Verifying update...") 
    print ("Uniaxial tension 1% strain")    
    hexa.update (0.5,steel,P0,U_UT)
    print ("Element Stress:")
    print (hexa.sig)
    print ("Calculated Stress:")
    print (str(207000*0.01)+" MPa")
    print ("Element Forces:")
    print (hexa.get_force(U_UT,P0))
    print ("Calculated Force:")
    print (str(207000*0.01*10*10)+" N")

    print ("\nSimple shear 1% strain")    
    hexa.update (0.5,steel,P0,U_SS)
    print ("Element Stress:")
    print (hexa.sig)
    print ("Element Forces:")
    print (hexa.get_force(U_UT,P0))
    print ("Calculated Force:")
    G = 207000 / (2.0*(1+0.3))
    gam = 0.1 / 10
    tau = G * gam
    FS = tau * 10 * 10
    print (str(FS)+" N")
    
    print ("Done HEX Element...")
    
    '''
    #SYSTEM TEST 1: Uniaxial tension test, single element
    p = st.strix()
    p.read_file("tests/SYSTEM_LEVEL_TEST/01_UT.strx")
    p.strix_explicit(p.Tf)
    del p
    '''
    
    '''
    #SYSTEM TEST 2: Simple shear, single element
    p = st.strix()
    p.read_file("tests/SYSTEM_LEVEL_TEST/02_SS.strx")
    p.strix_explicit(p.Tf)
    del p
    '''
    
    '''
    #SYSTEM TEST 3: Tet element test, uniaxial tension
    p = st.strix()
    p.read_file("tests/SYSTEM_LEVEL_TEST/03_TET.strx")
    p.strix_explicit(p.Tf)
    del p
    '''
    
    '''
    #SYSTEM TEST 4: Two element beam, connectivity
    p = st.strix()
    p.read_file("tests/SYSTEM_LEVEL_TEST/04_BEAM_2H.strx")
    p.strix_explicit(p.Tf)
    del p
    '''

if __name__ == "__main__":
    main()