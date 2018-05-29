import os
import torch
import numpy as np
import argparse
import random


parser = argparse.ArgumentParser(description='Generating binary strings')

parser.add_argument(
    '--output',
    type=str,
    default='./output',
    help='location of the data corpus')

parser.add_argument(
    '--length', type=int, default=50, help='length of generated sequences')

parser.add_argument(
    '--is_random',
    type=lambda x: (str(x).lower() == 'true'),
    default=False,
    help='random generate sequence shorter than length')

parser.add_argument(
    '--num_samples',
    type=int,
    default=1000,
    help='number of samples generated')

args = parser.parse_args()

    
def generate_binary(length=1):
    sum = 0
    s = ""
    for i in range(0, length):
        temp = random.randint(0, 1)
        sum += temp
        s += str(temp)
    return s + str(sum % 2)


if __name__ == "__main__":
    data = [
        generate_binary(
            length=args.length if args.is_random is False else random.randint(
                1, args.length + 1)) for i in range(0, args.num_samples)
    ]
    with open(args.output, 'w') as f:
        for l in data:
            f.write(l+'\n')
