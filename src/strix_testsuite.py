import sys
import strix as st

def main():
    
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