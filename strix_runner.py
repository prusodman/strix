import strix as st

def SS (p):
    p.bcs.append([1,1,0,1])
    p.bcs.append([1,2,0,1])
    p.bcs.append([1,3,0,1])
    
    p.bcs.append([2,1,0,1])
    p.bcs.append([2,2,0,1])
    p.bcs.append([2,3,0,1])
    
    p.bcs.append([3,1,0,1])
    p.bcs.append([3,2,0,1])
    p.bcs.append([3,3,0,1])
    
    p.bcs.append([4,1,0,1])
    p.bcs.append([4,2,0,1])
    p.bcs.append([4,3,0,1])
    
    p.bcs.append([5,2,1,2])
    p.bcs.append([6,2,1,2])
    p.bcs.append([7,2,1,2])
    p.bcs.append([8,2,1,2])

def UT (p):
    p.bcs.append([1,1,0,1])
    p.bcs.append([1,2,0,1])
    p.bcs.append([1,3,0,1])
    p.bcs.append([2,1,0,1])
    p.bcs.append([2,2,0,1])
    p.bcs.append([3,3,0,1])
    p.bcs.append([4,2,0,1])
    p.bcs.append([4,3,0,1])
    p.bcs.append([5,1,0,1])
    p.bcs.append([5,2,0,1])
    p.bcs.append([6,2,0,1])
    p.bcs.append([7,1,0,1])
    
    #move nodes
    p.bcs.append([5,3,1,2])
    p.bcs.append([6,3,1,2])
    p.bcs.append([7,3,1,2])
    p.bcs.append([8,3,1,2])
    
def main():
    #set up a problem
    p = st.strix()
    
    #node structure [nid,X,Y,Z]
    p.n0.append([1,0,0,0])
    p.n0.append([2,10,0,0])
    p.n0.append([3,10,10,0])
    p.n0.append([4,0,10,0])
    p.n0.append([5,0,0,10])
    p.n0.append([6,10,0,10])
    p.n0.append([7,10,10,10])
    p.n0.append([8,0,10,10])
    
    #   
    #
    #         7---------8
    #        /|        /|
    #       / |       / |
    #      /  |      /  |
    #     5---4-----6---3   
    #     |  /      |  /
    #     | /       | /
    #     |/        |/
    #     1---------2
    #
    #element structure [eid,N1,N2,N3,N4,N5,N6,N7,N8]
    p.elements.append(st.hex8([1,1,2,3,4,5,6,7,8]))
    
    SS(p)
    
    #bc structure [nid,dof,mag,typ]
    #typ = 1, disp
    #typ = 2, velocity
    #typ = 4, force
    # 1 - XD, 2 - YD, 3 - ZD
    # 4 - XR, 5 - YR, 6 - ZR
    
    p.strix_explicit(0.001,1,"Simple Shear Test Case")
    
    #print(p.elements[0].n0)
    #print(p.elements[0].n1)
    #print(p.elements[0].get_def_grad(0,0,0))
    
    print("\nFinal stress state:")
    print("====================")
    print(p.elements[0].sig)
    print("\n")
    
    print("\nFinal element force:")
    print("====================")
    print(p.elements[0].get_force(0,0,0))
    print("\n")
    
if __name__ == "__main__":
    main()