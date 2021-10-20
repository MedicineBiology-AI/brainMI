# -*- coding: utf-8 -*-
import os 
import argparse

import numpy as np
import scipy.stats as st

def set_args():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--N_path', 
                      default="../data/preprocessing/BFC_based_gene_network.npz", 
                      type=str, help='')
    parser.add_argument('--p', default=0.65,
                        type=float, help='')
    
    return parser.parse_args()


def filtered(matrix, p):
    matrix = np.tril(matrix, -1)
    matrix = matrix + matrix.T
    mask = (matrix > p).astype('int8')

    return mask


def main():
    args = set_args()
    
    files = np.load(args.N_path, allow_pickle=True)
    N, genes = files['N'], files['symbol']
    
    N = filtered(N, args.p)
    
    mask = N.sum(axis=1) > 0
    print("### Gene numbers: {}".format(mask.sum()))
    N = N[mask][:, mask]
    genes = genes[mask]
    print("### Edge number: {}".format(N.sum()))
    
    root_path, file_name = os.path.split(args.N_path)
    save_file_name = file_name.split('.')[0] + '_filtered.npz'
    np.savez(os.path.join(root_path, save_file_name), corr=N, symbol=genes)

if __name__ == "__main__":
    main()