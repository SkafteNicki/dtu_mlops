import sys

print(sys.argv)

if __name__ == "__main__":
    exp1 = sys.argv[1]
    exp2 = sys.argv[2]
    
    print(f'Comparing run {exp1} to {exp2}')