# EDDA
This is the source code of EDDA in CIKM'23. 

__Paper__: Multi-domain Recommendation with Embedding Disentangling and Domain Alignment (CIKM'23)


### Environment Setup

```
torch~=1.9.0+cu111
dgl-cu111~=0.7.1
numpy~=1.22.2
scipy~=1.7.1
tqdm~=4.62.3
```

### Guideline to run code
To run our EDDA
```
python .\run_edda.py --dataset ... --tasks ....
```

### Acknowledgements

Part of our codes in `LibMTL/` folder are from [LibMTL](https://github.com/median-research-group/LibMTL).

If you use our datasets or codes, please cite our paper.
```
@inproceedings{EDDA,
    author = {Ning, Wentao and Yan, Xiao and Liu, Weiwen and Cheng, Reynold and Zhang, Rui and Tang, Bo},
    title = {Multi-domain Recommendation with Embedding Disentangling and Domain Alignment},
    booktitle = {CIKM},
    publisher = {ACM},
    year = {2023}
}
```
