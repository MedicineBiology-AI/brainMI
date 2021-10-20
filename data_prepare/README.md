# data prepare

First, download and install workbench from https://www.humanconnectome.org/software/get-connectome-workbench. 

Then, 
```bash 
# 1. Obtain BFC-based similarity matrix
python BFC_based_gene_network_construction.py

# 2. Obtain BFC-based gene network (binarized)
python BFC_based_gene_network_filter.py

# 3. Obtain database-based ppi network and exprimental-based ppi network
python get_ppi_sub_net.py

# 4. Obtain brain disease genes 
python get_brainDiseasegenes.py
```

