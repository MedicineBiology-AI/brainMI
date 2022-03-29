# -*- coding: utf-8 -*-
import os
import logging
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import nibabel as nib
import torch


def set_args():
    parser = argparse.ArgumentParser(description='Build gene network')
    parser.add_argument('--expression_path', 
                        default="../data/raw/humanbrainmap/normalized_microarray_donor10021/MicroarrayExpression.csv", 
                        type=str, help="The path to probe expression file.")
    parser.add_argument('--sampleAnnot',
                        default="../data/raw/humanbrainmap/normalized_microarray_donor10021/SampleAnnot.csv",
                        type=str, help="The path to sample annotation file.")
    parser.add_argument('--sampleAnnot_microarray',
                        default="../data/raw/humanbrainmap/normalized_microarray_donor10021/SampleAnnot.csv",
                        type=str, help="The path to mcroarray sample annotation file.")
    parser.add_argument('--template_path',
                        default="../data/raw/cole/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii",
                        type=str, help='The path to template file.')
    parser.add_argument('--r_sur_gii_path',
                    default="../data/raw/cole/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii",
                    type=str, help="The path to left surface gii file.")
    parser.add_argument('--l_sur_gii_path',
                        default="../data/raw/cole/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii",
                        type=str, help="The path to right surface gii file.")
    parser.add_argument('--tmp_path',
                        default='../data/preprocessing',
                        type=str, help="The path to tmp path.")
    parser.add_argument('--cole_root_path', 
                        default="../data/raw/cole/",
                        type=str, help='https://doi.org/10.1016/j.neuroimage.2018.10.006')
    parser.add_argument('--output',
                        default='../data/preprocessing',
                        # default = '../data/preprocessing/microarray10021/',
                        type=str, help="BFC-based gene network construction path.")
    parser.add_argument('--probe_path', default="../data/raw/humanbrainmap/normalized_microarray_donor10021/Probes.csv",
                        type=str, help='')
    
    return parser.parse_args()


def identify_specific_genes(path, B):
    df = pd.read_csv(path, header=None, index_col=0)
    
    probs, expression_matrix = df.index, df.values
    genes = B.index
    expression_matrix = (np.dot(B.values, expression_matrix).T / B.values.sum(axis=1)).T
    
    normalized_expression_matrix = (expression_matrix - np.mean(expression_matrix, axis=1, keepdims=True)) / (np.std(expression_matrix, axis=1, keepdims=True) + 1e-8)
    
    return genes, normalized_expression_matrix


def obtain_FCmat(root_path):
    
    def obtain_grayordinate_label(parcelCIFTIFile):
        img = nib.load(parcelCIFTIFile)
        data = np.array(img.dataobj).squeeze().astype('int16')
        
        return data
    
    parcelCIFTIFile = os.path.join(root_path, 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii')

    parcelTSFilename = os.path.join(root_path, 'Output_Atlas_CortSubcort.Parcels.LR.ptseries.nii')

    #Set this to be your input fMRI data CIFTI file
    #inputFile='Run1_fMRIData_Atlas.dtseries.nii'
    inputFile = "../data/raw/humanconnectome/HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500_Eigenmaps.dtseries.nii"

    # Load in dense array time series using nibabel
    dtseries = np.squeeze(nib.load(inputFile).get_fdata())
    # Find number of time points
    n_timepoints = dtseries.shape[1]

    # Parcellate dense time series using wb_command
    os.system('wb_command -cifti-parcellate ' + inputFile + ' ' + parcelCIFTIFile + ' COLUMN ' + parcelTSFilename + ' -method MEAN')

    # Load in parcellated data using nibabel
    lr_parcellated = np.squeeze(nib.load(parcelTSFilename).get_fdata()).T

    # Loading community ordering files
    netassignments = np.loadtxt(os.path.join(root_path, 'cortex_subcortex_parcel_network_assignments.txt'))
    # need to subtract one to make it compatible for python indices
    indsort = np.loadtxt(os.path.join(root_path, 'cortex_subcortex_community_order.txt'),dtype=int) - 1 
    indsort.shape = (len(indsort),1)

    # Computing functional connectivity and visualizing the data (assuming preprocessing has already been done)
    FCmat = np.corrcoef(lr_parcellated)
    FCMat_sorted = FCmat[indsort,indsort.T]
    
    grayordinate_label = obtain_grayordinate_label(parcelCIFTIFile) - 1
    

    return FCMat_sorted, grayordinate_label

def get_region_coords(sampleAnnot, sampleAnnot_mcroarray):
    df_1, df = pd.read_csv(sampleAnnot), pd.read_csv(sampleAnnot_mcroarray)
    # df_2.set_index(['well_id'], inplace=True)
    # df = df_2.loc[df_1['well_id']]
    # df.reset_index(inplace=True)

    indexes, results = [], []
    for i in range(len(df)):
        result = (df['mni_x'][i], df['mni_y'][i], df['mni_z'][i])
        results.append(result)

        if df['slab_type'][i] == 'CX':
            if 'right' in df['structure_name'][i]:
                indexes.append(df['slab_type'][i] + '_R')
            else:
                indexes.append(df['slab_type'][i] + '_L')
        else:
            indexes.append(df['slab_type'][i])

    return indexes, results


class brainGrayordinateCoords():
    def __init__(self, template_path, grayordinate_label):
        self.grayordinate_label = grayordinate_label
        template = nib.load(template_path)
        header = template.header
        self.grayord_maps = header.get_index_map(1)
        self.ijk2xyz_matrix = self.grayord_maps[0].transformation_matrix_voxel_indices_ijk_to_xyz.matrix
        
        # grayord maps一共有21 maps, 包括1-->cx_l, 2-->cx_r, 7-->bs, 10, 11-->cb
        self.indexOfsets = [self.coords_counts(1), self.coords_counts(2),
                            self.coords_counts(7), self.coords_counts(10)]
    
    def coords_counts(self, n):
        counts = 0
        for i in range(1, n):
            try:
                counts += len([j for j in self.grayord_maps[i].vertex_indices])
            except:
                counts += len([j for j in self.grayord_maps[i].voxel_indices_ijk])
                
        return counts
        
    def transform(self, matrix):
        matrix_ = np.insert(matrix, 3, np.ones(matrix.shape[0]), axis=1)
        tmp = np.dot(self.ijk2xyz_matrix, matrix_.T).T
        tmp = np.delete(tmp, -1, axis=1)
        
        return tmp
    
    def get_coords(self):
        cxl_vertex = {j: i for i, j in enumerate(self.grayord_maps[1].vertex_indices)}
        cxr_vertex = {j: i for i, j in enumerate(self.grayord_maps[2].vertex_indices)}
        
        bs_voxel = np.array([i for i in self.grayord_maps[7].voxel_indices_ijk])
        cb_voxel = np.array([i for i in self.grayord_maps[10].voxel_indices_ijk] + [i for i in self.grayord_maps[11].voxel_indices_ijk])
        bs_voxel, cb_voxel = self.transform(bs_voxel), self.transform(cb_voxel)
        
        return cxl_vertex, cxr_vertex, bs_voxel, cb_voxel
    
    def mapping_region2grayordinate(self, sampleAnnot, sampleAnnot_mcroarray, 
                                    l_sur_gii_path, r_sur_gii_path, tmp_path):
        
        slab_types, region_coords = get_region_coords(sampleAnnot, sampleAnnot_mcroarray)
        
        cxl_vertex, cxr_vertex, bs_voxel, cb_voxel = self.get_coords()
        
        region_grayordinate_relationship = np.zeros((len(slab_types), len(set(list(self.grayordinate_label)))))
        
        for i, (slab_type, coord) in tqdm(enumerate(zip(slab_types, region_coords))):
            with open(os.path.join(tmp_path, 'tmp.txt'), 'w') as f:
                for line in coord:
                    f.write(str(line) + " ")

            if slab_type.startswith('CX_L'):
                os.system('wb_command -surface-closest-vertex'
                          + ' ' + l_sur_gii_path
                          + ' ' + os.path.join(tmp_path, 'tmp.txt')
                          + ' ' + os.path.join(tmp_path, 'output.txt'))
                with open(os.path.join(tmp_path, 'output.txt'), 'r') as f:
                    vertex = [int(f.read())]
                    
                if vertex[0] in cxl_vertex.keys():
                    index = cxl_vertex[vertex[0]]
                    region_grayordinate_relationship[i, self.grayordinate_label[index]] = 1
            
            elif slab_type == 'CX_R':
                os.system('wb_command -surface-closest-vertex'
                          + ' ' + r_sur_gii_path
                          + ' ' + os.path.join(tmp_path, 'tmp.txt')
                          + ' ' + os.path.join(tmp_path, 'output.txt'))
                with open(os.path.join(tmp_path, 'output.txt'), 'r') as f:
                    vertex = [int(f.read())]

                if vertex[0] in cxr_vertex.keys():
                    index = cxr_vertex[vertex[0]] + self.indexOfsets[1]
                    region_grayordinate_relationship[i, self.grayordinate_label[index]] = 1
                        
            elif slab_type == 'BS':
                distances = np.sum(np.square(bs_voxel - coord), axis=1)
                # if np.min(distances) <= -2: # only consider CX, not BS
                if np.min(distances) <= 1:
                    index = np.squeeze(np.where(distances == np.min(distances))) + self.indexOfsets[2]
                    region_grayordinate_relationship[i, self.grayordinate_label[index]] = 1
                    
            elif slab_type == 'CB':
                distances = np.sum(np.square(cb_voxel - coord), axis=1)
                # if np.min(distances) <= -2: # only consider CX, not CB
                if np.min(distances) <= 1:
                    index = np.squeeze(np.where(distances == np.min(distances))) + self.indexOfsets[3]
                    region_grayordinate_relationship[i, self.grayordinate_label[index]] = 1
        
        return region_grayordinate_relationship            
        

def weight_cup(cup, P):
    # result = torch.sum(torch.matmul(cup, P) * cup, dim=1)
    # result = torch.sum(torch.matmul(cup, P), dim=1)
    
    result = result = torch.sum(torch.matmul(cup, P) * cup, dim=1)

    return result


def obtain_N1(Q, P):
    N_2 = np.zeros((Q.shape[0], Q.shape[0]))

    P = torch.from_numpy(P).float().cuda().detach()
    Q = torch.from_numpy(Q).float().cuda().detach()
    
    a, b = torch.tensor(1.0).cuda().detach(), torch.tensor(1.0).cuda().detach()

    for i in tqdm(range(N_2.shape[0])):
        # cup = ((Q[i] + Q) > 0).float()
        dif = a / (b + torch.abs(Q[i] - Q))
        # dif = torch.abs(Q[i] - Q)
        N_2[i] = weight_cup(dif, P).cpu().numpy()

    return N_2


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def genes_probes_relationship(probes_path):
    """获取probe的gene标签（one-hot label,）

    Args:
        probes_path (str): The path to probes annotation file.

    Return:
        DataFrame: probes to genes one-hot matrix. Index is gene and column is probe.
    """
    df = pd.read_csv(probes_path)
    probe_list = df["probe_id"].tolist()
    # gene_list = list(set(df["gene_symbol"]))
    gene_list = []
    for i in df['gene_symbol']:
        if i not in gene_list:
            gene_list.append(i)

    dict_g_p = {j: i for i, j in enumerate(gene_list)}
    probe_lables = [dict_g_p[i] for i in df["gene_symbol"]]

    matrix = dense_to_one_hot(np.array(probe_lables), len(gene_list))
    matrix_df = pd.DataFrame(matrix.T, index=gene_list, columns=probe_list)

    return matrix_df

def main():
    args = set_args()
    logging.basicConfig(level=logging.INFO)
    
    B = genes_probes_relationship(args.probe_path)
    
    # standardized gene expression 
    logging.info('Start to standardize gene expression data')
    genes, mask_high = identify_specific_genes(args.expression_path, B)
    logging.info('End standardized gene expression data')
    
    # parcel-based brain functional connectivity
    logging.info('Start to build parcel-based brain functional connectivity')
    FCmat_parcel, grayordinate_label = obtain_FCmat(args.cole_root_path)
    logging.info('End building parcel-based brain functional connectivity')
    
    # construct brain region-parcel relationship matrix
    logging.info('Start to construct brain region-parcel relationship matrix')
    grayordinate = brainGrayordinateCoords(args.template_path, grayordinate_label)
    region_grayordinate_relationship = grayordinate.mapping_region2grayordinate(args.sampleAnnot,
                                                                                args.sampleAnnot_microarray,
                                                                                args.l_sur_gii_path,
                                                                                args.r_sur_gii_path,
                                                                                args.tmp_path)
    # region_grayordinate_relationship = np.load(os.path.join(args.output, "BFC_based_network.npz"), allow_pickle=True)['region_parcel_relation']
    logging.info('End constructing brain region-parcel relationship matrix')
    
    # obtain brain-region-based brain functional connectivity
    logging.info('Start to construct brain-region-based brain functional connectivity')
    FCmat_brain_regions = np.dot(np.dot(region_grayordinate_relationship, FCmat_parcel), region_grayordinate_relationship.T)
    FCmat_brain_regions = np.abs(np.tanh(FCmat_brain_regions))
    FCmat_brain_regions = FCmat_brain_regions - np.diag(np.diag(FCmat_brain_regions))
    logging.info('End constructing brain-region-based brain functional connectivity')
    
    # construct BFC-based gene network
    logging.info('Start to construct BFC-based gene network')
    N1_high = obtain_N1(mask_high, FCmat_brain_regions)
    N2_high = FCmat_brain_regions.sum()
    N_high = N1_high / (N2_high + 1e-8)
    logging.info('End constructing BFC-based gene network')
    
    np.savez(os.path.join(args.output, "BFC_based_network.npz"), N=N_high, symbol=genes,
             region_parcel_relation=np.array(region_grayordinate_relationship))

if __name__ == "__main__":
    main()
