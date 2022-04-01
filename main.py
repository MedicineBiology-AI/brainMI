import os 
import argparse


id2disease = {
    "10652": "Alzheimer's disease",
    "14330": "Parkinson disease",
    "1470": "Major depressive disorder",
    "12849": "Autism"
}


def set_args():
    parser = argparse.ArgumentParser(description="Train the model to obtain features for The next process")
    parser.add_argument('--cfg_paths', type=str, default=['config/config_embedding_onlyForBFC.json', 'config/config_embedding_integration.json'], 
                        nargs='+', help='The configs to train model.')
    parser.add_argument('--save_paths', type=str, default=['data/output/onlyforBFC_gene_net', 'data/output/integration'], 
                        nargs='+', help="The path to save model and result. ")
    parser.add_argument('--disease_gene_paths', type=str, default=[
        "data/preprocessing/DOID_10652_genes.npy", "data/preprocessing/DOID_14330_genes.npy",
        "data/preprocessing/DOID_1470_genes.npy", "data/preprocessing/DOID_12849_genes.npy"], 
                        nargs='+',  help=' ')

    args = parser.parse_args()

    return args


def obtain_gene_features(cfg_path, save_path):
    os.system('python  net_embedding.py'+' --cfg_path '+cfg_path+' --save_path '+save_path)

def evalution(save_path, disease_genes):
    files = os.listdir(save_path)
    files = [os.path.join(save_path, i) for i in files if i.endswith('.npz')]
    arg = " ".join(files)

    os.system('python train_svm.py'+' --y_path '+disease_genes+' --x_path '+arg)

def main():
    args =  set_args()

    for cfg_path, save_path in zip(args.cfg_paths, args.save_paths):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            obtain_gene_features(cfg_path, save_path)

    for save_path in args.save_paths:
        if save_path.endswith('onlyforBFC_gene_net'):
            print('########################################################')
            print('### Evaluate BFC-based gene network.')
        else:
            print('########################################################')
            print('### Evaluate brainMI by integrating PPIs and BFC-based gene network.')

        for disease_gene_path in args.disease_gene_paths:
            disease_id = disease_gene_path.split('_')[1]
            print('### For predicting {} genes.'.format(id2disease[disease_id]))

            evalution(save_path, disease_gene_path)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print('########################################################')



if __name__ == "__main__":
    main()
