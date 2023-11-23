# Multi-gate Mixture of Multi-view Graph Contrastive Learning on Electronic Health Record

This is a PyTorch reference implementation of the MMMGCL in "Multi-gate Mixture of Multi-view Graph Contrastive Learning on Electronic Health Record" (Yu Cao, Qian Wang, Xu Wang, Dezhong Peng, Peilin Li, JBHI) [Paper](https://pubmed.ncbi.nlm.nih.gov/37851554/)

## Usage

+ Code for MIMIC-III and eICU pre-processing in `data_pre`
+ Code for glove pre-training in `data_pre/glove/glove_torch.py`
+ Code for running experiments in `multi_task/multi_task.py`

## Citations
```
@ARTICLE{10287395,
  author={Cao, Yu and Wang, Qian and Wang, Xu and Peng, Dezhong and Li, Peilin},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Multi-gate Mixture of Multi-view Graph Contrastive Learning on Electronic Health Record}, 
  year={2023},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/JBHI.2023.3325221}}
```