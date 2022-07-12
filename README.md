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
```
@InProceedings{lovenia2022ascend,
  author    = {Lovenia, Holy  and  Cahyawijaya, Samuel  and  Winata, Genta  and  Xu, Peng  and  Xu, Yan  and  Liu, Zihan  and  Frieske, Rita  and  Yu, Tiezheng  and  Dai, Wenliang  and  Barezi, Elham J.  and  Chen, Qifeng  and  Ma, Xiaojuan  and  Shi, Bertram  and  Fung, Pascale},
  title     = {ASCEND: A Spontaneous Chinese-English Dataset for Code-switching in Multi-turn Conversation},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {7259--7268},
  abstract  = {Code-switching is a speech phenomenon occurring when a speaker switches language during a conversation. Despite the spontaneous nature of code-switching in conversational spoken language, most existing works collect code-switching data from read speech instead of spontaneous speech. ASCEND (A Spontaneous Chinese-English Dataset) is a high-quality Mandarin Chinese-English code-switching corpus built on spontaneous multi-turn conversational dialogue sources collected in Hong Kong. We report ASCEND's design and procedure for collecting the speech data, including annotations. ASCEND consists of 10.62 hours of clean speech, collected from 23 bilingual speakers of Chinese and English. Furthermore, we conduct baseline experiments using pre-trained wav2vec 2.0 models, achieving a best performance of 22.69\% character error rate and 27.05\% mixed error rate.},
  url       = {https://aclanthology.org/2022.lrec-1.788}
}

```
