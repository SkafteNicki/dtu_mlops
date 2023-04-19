import argparse
import os
from multiprocessing import Pool

def script_launcher(script):
    os.system(f"python {script}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str)
    parser.add_argument('-num_parallel', default=2, type=int)
    args = parser.parse_args()
    
    print(f"Arguments: {args.__dict__}")
    
    pool = Pool(processes=args.num_parallel) 
    pool.map(script_launcher, [args.script for _ in range(args.num_parallel)]) 