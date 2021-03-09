# SimTriplet
SimTriplet: PyTorch Implementation

## Data description 
Image data used for model pretrain, finetune and test can be downloaded via Google drive link:

Train: [link](https://drive.google.com/drive/folders/14Cg-QuOCPVrynpuFI_jFqRqzTj2rNk4d?usp=sharing)  
Finetune: [link](https://drive.google.com/drive/folders/1-XaRXqBOrAHQNyMNEBwCEsdKilk_JFkz?usp=sharing)  
Test: [link](https://drive.google.com/drive/folders/1Hpvo2iNqt3I1qgMy9SCXv7azSpCEitao?usp=sharing)

## Pre-train SimTriplet 
main_tcga_mixpresition.py

``` python
--data_dir
../Data/
--log_dir
../logs/
-c
configs/TCGA_triple.yaml
--ckpt_dir
./checkpoint/
```

## Finetune linear classifier
TCGA_linear_cross_val.py


## Test
test_patho_prob_major_vote.py

## Well trained model
Model pretrained on TCGA data is provided [here](https://drive.google.com/file/d/1TtiMckXEjBV17UICQ1tpjSATP8u4fLOA/view?usp=sharing)

