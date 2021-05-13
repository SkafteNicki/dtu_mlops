from multiprocessing import Pool
import argparse
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str)
    parser.add_argument('-num_parallel', default=2, type=int)
    args = parser.parse_args()
    
    print(f"Arguments: {args.__dict__}")
    
    def script_launcher():
        os.system(f"python {args.script}")      
    
    pool = Pool(processes=args.num_parallel) 
    pool.map(script_launcher, list(range(args.num_parallel))) 