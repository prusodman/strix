import sys
import strix as st

def main():
    fname = sys.argv[1]
    
    #set up a problem
    p = st.strix()
    p.read_file(fname)
    
    p.strix_explicit(p.Tf)

if __name__ == "__main__":
    main()