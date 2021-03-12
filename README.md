# SimTriplet
SimTriplet: PyTorch Implementation
[Paper](https://arxiv.org/pdf/2103.05585.pdf)

![Visualization of classification](https://github.com/hrlblab/SimTriplet/blob/main/For_github.jpg)

## Data description 
Image data used for model pretrain, finetune and test can be downloaded via Google drive link:

Train: [link](https://drive.google.com/drive/folders/14Cg-QuOCPVrynpuFI_jFqRqzTj2rNk4d?usp=sharing)  
Finetune: [link](https://drive.google.com/drive/folders/1-XaRXqBOrAHQNyMNEBwCEsdKilk_JFkz?usp=sharing)  
Test: [link](https://drive.google.com/drive/folders/1Hpvo2iNqt3I1qgMy9SCXv7azSpCEitao?usp=sharing)  

The complete training and testing datasets will be provided once the paper is officially published.

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

## Citation
If you find this repository useful in your research, please cite:
```
@misc{liu2021simtriplet,
      title={SimTriplet: Simple Triplet Representation Learning with a Single GPU}, 
      author={Quan Liu and Peter C. Louis and Yuzhe Lu and Aadarsh Jha and Mengyang Zhao and Ruining Deng and Tianyuan Yao and Joseph T. Roland and Haichun Yang and Shilin Zhao and Lee E. Wheless and Yuankai Huo},
      year={2021},
      eprint={2103.05585},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```

