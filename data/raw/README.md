# Data
The downloaded raw data is organized as follows: 
```
├─humanconnectome
│  └─HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500_Eigenmaps.dtseries.nii
├─cole
│  ├─S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii
│  ├─cortex_subcortex_community_order.txt
│  ├─S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii
│  ├─cortex_subcortex_parcel_network_assignments.txt
│  └─CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii
├─humanbrainmap
│  └─normalized_microarray_donor10021
│     ├─PACall.csv
│     ├─Probes.csv
│     ├─MicroarrayExpression.csv
│     ├─Readme.txt
│     ├─SampleAnnot.csv
│     └─Ontology.csv
├─README.md
├─9606.protein.info.v11.0.txt
├─HumanDO.obo
├─9606.protein.links.detailed.v11.0.txt
└─all_gene_disease_associations.tsv
```

- cole/ -- https://github.com/ColeLab/ColeAnticevicNetPartition [^1]
- humanbrainmap/normalized_microarray_donor10021/ -- https://human.brain-map.org/api/v2/well_known_file_download/178238373 [^2]
- humanconnectome/ -- https://db.humanconnectome.org/app/action/ChooseDownloadResources?project=HCP_Resources&resource=GroupAvg&filePath=HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500_Eigenmaps.dtseries.zip [^3]
- 9606.protein.info.v11.0.txt and 9606.protein.links.detailed.v11.0.txt -- https://string-db.org/cgi/download?sessionId=%24input-%3E%7BsessionId%7D&species_text=Homo+sapiens [^4]
- all_gene_disease_associations.tsv -- https://www.disgenet.org/downloads [^5]
- HumanDo.obo -- https://github.com/DiseaseOntology/HumanDiseaseOntology/tree/main/src/ontology [^6]


[^1]: Ji, Jie Lisa, et al. "Mapping the human brain's cortical-subcortical functional network organization." Neuroimage 185 (2019): 35-57.  
[^2]: Shen, Elaine H., Caroline C. Overly, and Allan R. Jones. "The Allen Human Brain Atlas: comprehensive gene expression mapping of the human brain." Trends in neurosciences 35.12 (2012): 711-714.  
[^3]: Sha, Zhiqiang, et al. "Common dysfunction of large-scale neurocognitive networks across psychiatric disorders." Biological psychiatry 85.5 (2019): 379-388.  
[^4]: Szklarczyk, Damian, et al. "STRING v10: protein–protein interaction networks, integrated over the tree of life." Nucleic acids research 43.D1 (2015): D447-D452.   
[^5]: Piñero, Janet, et al. "The DisGeNET knowledge platform for disease genomics: 2019 update." Nucleic acids research 48.D1 (2020): D845-D855.  
[^6]: Schriml, Lynn M., et al. "Human Disease Ontology 2018 update: classification, content and workflow expansion." Nucleic acids research 47.D1 (2019): D955-D962.  

