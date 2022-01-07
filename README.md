# ASCEND

ASCEND (A Spontaneous Chinese-English Dataset) introduces a high-quality resource of spontaneous multi-turn conversational dialogue Chinese-English code-switching corpus collected in Hong Kong. 

### Download the dataset
You can find ASCEND at [HuggingFace](https://huggingface.co/datasets/CAiRE/ASCEND)

Download ASCEND
```
git lfs install
git clone https://huggingface.co/datasets/CAiRE/ASCEND
# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

### Cite us
@misc{lovenia2021ascend,
      title={ASCEND: A Spontaneous Chinese-English Dataset for Code-switching in Multi-turn Conversation}, 
      author={Holy Lovenia and Samuel Cahyawijaya and Genta Indra Winata and Peng Xu and Xu Yan and Zihan Liu and Rita Frieske and Tiezheng Yu and Wenliang Dai and Elham J. Barezi and Pascale Fung},
      year={2021},
      eprint={2112.06223},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
