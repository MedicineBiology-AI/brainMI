import os 
import argparse

import numpy as np
import pandas as pd
import networkx as nx

def set_args():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--PPI_info_path', 
                      default="../data/raw/9606.protein.info.v11.0.txt", 
                      type=str, help='')
    parser.add_argument('--PPI_net_path', default="../data/raw/9606.protein.links.detailed.v11.0.txt",
                        type=str, help='')
    parser.add_argument('--out_path', default="../data/preprocessing",
                        type=str, help='')
    
    return parser.parse_args()

def remove_isolated_point(net, node):
    net = np.tril(net, -1)
    net = net + net.T

    mask = (net > 0).sum(axis=1) != 0

    return node[mask], net[mask][:, mask]

args = set_args()

dg = pd.read_csv(args.PPI_info_path, sep='\t')
ensp2hgnc = {i: j for i, j in dg[['protein_external_id', 'preferred_name']].itertuples(index=False, name=None)}

df = pd.read_csv(args.PPI_net_path, sep=' ')

# df.query('coexpression > 0', inplace=True)

df['protein1'] = df['protein1'].apply(lambda x: ensp2hgnc[x])
df['protein2'] = df['protein2'].apply(lambda x: ensp2hgnc[x])

SUB_NAMES = ['experimental', 'database']
for SUB_NAME in SUB_NAMES: 
    nt = nx.from_pandas_edgelist(df, 'protein1', 'protein2', SUB_NAME)

    mat = nx.to_numpy_matrix(nt, weight=SUB_NAME)

    mat = mat.astype('int16')

    nodes = np.array(nt.nodes)

    nodes, mat = remove_isolated_point(mat, nodes)

    np.savez(os.path.join(args.out_path, 'ppi_subnetwork_'+str(SUB_NAME)+'.npz'), corr=mat, symbol=nodes)
