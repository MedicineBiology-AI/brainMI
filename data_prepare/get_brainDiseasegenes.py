import argparse
import os

from pronto import Ontology
import pandas as pd
import numpy as np
import re


def set_args():
    parser = argparse.ArgumentParser(description='Obtain brain disease association genes')
    parser.add_argument('--obo_path', default='../data/raw/HumanDO.obo',
                        type=str, help='The path to disease ontology file.')
    parser.add_argument('--gda_path', default='../data/raw/all_gene_disease_associations.tsv',
                        type=str, help='The path to gene_disease association file.')
    parser.add_argument('--output', default='../data/preprocessing/',
                        type=str, help='Tha output path.')

    args = parser.parse_args()

    return args


def sub_diseases(obo_path: str, dise_id: str):
    go = Ontology(obo_path)
    instruments = set(go[dise_id].subclasses())

    data = []
    for term in instruments:
        value = {"id": int(term.id[5:]), "name": term.id, "desc": term.name, "xrefs": term.xrefs}
        # go["is_a"]
        parents = instruments.intersection(term.relationships.get(go.get_relationship("is_a"), set()))

        if parents:
            value["parent"] = int(parents.pop().id[5:])
        data.append(value)

    return data


def main():
    args = set_args()

    obo_path = args.obo_path
    # dise_ids = ['DOID:14330']  # parkinson
    # dise_ids = ['DOID:10652']  # Alzheimer's disease
    # dise_ids = ['DOID:1470']  # MDD
    # dise_ids = ['DOID:12849']  #	autistic disorder

    dise_ids = ['DOID:14330', 'DOID:10652', 'DOID:1470', 'DOID:12849']


    # dise_ids = ['DOID:1826']  # 癫痫
    # dise_ids = ['DOID:6713']  # 	脑血管疾病

    for dise_id in dise_ids:

        data = sub_diseases(obo_path, dise_id)

        # Save disease ids and disease names
        diseaseIds = []
        diseaseNames = []
        for value in data:
            diseaseNames.append(value["desc"])

            str_list = map(str, list(value["xrefs"]))
            str_value = "".join(str_list)

            result = re.findall(r"UMLS_CUI:[A-Za-z0-9]*", str_value)
            if result:
                diseaseIds.extend([i.split(':')[-1] for i in result])

        df = pd.read_csv(args.gda_path, sep="\t", header=0)
        df_with_row_name = df[df["diseaseId"].isin(diseaseIds) | df["diseaseName"].isin(diseaseNames)]

        id_name = dise_id.split(':')[-1]
        genes = np.array(df_with_row_name['geneSymbol'].tolist())
        np.save(os.path.join(args.output, 'DOID_' + id_name + '_genes.npy'), genes)


if __name__ == "__main__":
    main()
