# CNAttention: an attention-based deep multiple-instance method for uncovering CNA signatures across cancers

This tool aims to uncovering copy number abbresion (CNA) patterns of pan-cancer based on large CNA data from progenetix database.

![pipeline_diagram](https://github.com/ziyingyang96/cnv-signature/blob/main/workflow.png)


# Installation
Firstly you have to make sure that you have all dependencies in place. The simplest way to do so, is to use anaconda.

You can create an anaconda environment called cnattention.

```    
conda env create -f requirements.txt
conda activate cnattention
```

# Running

`python CNAttention.py`

The script `CNAttention.py` takes the CNA profiles as input and output the attention parameters for all CNA features, accuracy of bag and instance classification. The users can adjust the number of selected features as the final pattern of different cancers.


# Visualization

The users can visualize the selected CNA patterns on [progenetix](https://progenetix.org/) database by simply type in the feature gene names of a specific cancer type.

