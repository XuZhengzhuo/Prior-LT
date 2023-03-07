## Towards Calibrated Model for Long-Tailed Visual Recognition from Prior Perspective
Zhengzhuo Xu, Zenghao Chai, Chun Yuan
_________________

This is the PyTorch implementation of our [paper](https://openreview.net/forum?id=vqzAfN-BoA_&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2021%2FConference%2FAuthors%23your-submissions)) in NeurIPS 2021.

### Environment settings
- PyTorch >= 1.4
- Scikit-learn
- Matplotlib

### Training

To train the model, please select a config file path or customize by yourself. For example:

```powershell
python train.py config/cifar10_100.py
```
The result will be saved in `./result`.


### Inferance model

| Dataset          | log                                                                                        | Model                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| CIFAR-10-LT-50   | [link](https://drive.google.com/file/d/1JHxk-qMM1HMDQtduhq9KWoZfZpzxgMRO/view?usp=sharing) | [link](https://drive.google.com/file/d/1tclscVkcXj0lJum7Azy8qHecB7Pomc0c/view?usp=sharing) |
| CIFAR-10-LT-100  | [link](https://drive.google.com/file/d/1YxfY_YEKHEwflyEB75_Bi8y2V0Gvoz21/view?usp=sharing) | [link](https://drive.google.com/file/d/1f8tNEBNoUarsi-I0LfdTuvMOwggAnuat/view?usp=sharing) |
| CIFAR-10-LT-200  | [link](https://drive.google.com/file/d/12uU7PPHOqYeoZqQ3u-yjG5VZ22Nq6-71/view?usp=sharing) | [link](https://drive.google.com/file/d/1GTf42bpfDmMz5MHTVsX9YkjLSeo9WJ-v/view?usp=sharing) |
| CIFAR-100-LT-50  | [link](https://drive.google.com/file/d/1TBHHl_VSDNakG32XOW9rV2Ebss5tum9B/view?usp=sharing) | [link](https://drive.google.com/file/d/1PKpxeeCO5ZRAq4srleTlcQqTTjQd6JfT/view?usp=sharing) |
| CIFAR-100-LT-100 | [link](https://drive.google.com/file/d/1cn0xdE5VxBb6ASAxZefO7PDjE8gfozlg/view?usp=sharing) | [link](https://drive.google.com/file/d/1SLowEae9vp3gWTFVKcjVnyb57iXH0yBt/view?usp=sharing) |
| CIFAR-100-LT-200 | [link](https://drive.google.com/file/d/1HBdY2Dlwh_kSFzJe9r6SWZDdlgIALYq8/view?usp=sharing) | [link](https://drive.google.com/file/d/16JUoxnbxuO7nivjw4M0LkUQiJ9AyAJDm/view?usp=sharing) |

### Note
- Some settings may be a little different from the paper reported. Because we make further optimization considering reviewers' suggestions.
- Some bugs need to be fixed when imb factor = 0.1
- Need to solve the test-agnostic situation.


### Reference

Welcome to cite:
```bib
@inproceedings{PriorLT,
    title={Towards Calibrated Model for Long-Tailed Visual Recognition from Prior Perspective},
    author={Xu, Zhengzhuo and Chai, Zenghao and Yuan, Chun},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year={2021}
}
```

### Acknowledgement
We adopt the code from the following repos. We thank them for providing their awesome code.
- [LDAM-DRW](https://github.com/kaidic/LDAM-DRW)
- [Bag-of-Tricks](https://github.com/zhangyongshun/BagofTricks-LT)
