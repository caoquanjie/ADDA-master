import argparse
import os
import test
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']='1'


def main():
    parser = argparse.ArgumentParser(description="manual to this script")
    parser.add_argument('--step',type=int,default=2)

    args = parser.parse_args()
    if args.step == 1:
        test.step1()
        return
    elif args.step == 2:
        test.step2()
        return
    elif args.step == 3:
        test.step3()
        return

if __name__ == "__main__":
    main()
