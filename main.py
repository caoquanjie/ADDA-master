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
        train.step1()
        return
    elif args.step == 2:
        train.step2()
        return
    elif args.step == 3:
        train.step3()
        return

if __name__ == "__main__":
    main()
