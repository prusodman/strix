import strix as st

def main():
    #set up a problem
    p = st.strix()
    p.read_file("sample_decks/SS.strx")
    #node structure [nid,X,Y,Z]
    #element structure [eid,N1,N2,N3,N4,N5,N6,N7,N8]
    
    #bc structure [nid,dof,mag,typ]
    #typ = 1, disp
    #typ = 2, velocity
    #typ = 4, force
    # 1 - XD, 2 - YD, 3 - ZD
    # 4 - XR, 5 - YR, 6 - ZR
    
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
    
    p.strix_explicit(0.00001,0.01)
    
    print("\nFinal nodal coord:")
    print("====================")
    print(p.elements[0].n1)
    print("\n")
    
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