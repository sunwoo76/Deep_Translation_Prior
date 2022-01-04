# Official implementation of DTP

This is the official implementation of our paper :

**Deep Translation Prior: Test-time Training for Photorealistic Style Transfer(AAAI 2022)**

Authors: Sunwoo Kim, Soohyun Kim and Seungryong Kim

You can check out the paper on [[arXiv](https://arxiv.org/abs/2112.06150)].

# Network

Our model DTP is illustrated below:

![alt text](/images/network.png)

## Example Results
<center><img src="images/gif/result1.gif" width="160px" class="center"/> <img src="images/gif/result2.gif" width="160px" class="center"/> <img src="images/gif/result4.gif" width="160px" class="center"/> <img src="images/gif/result5.gif" width="160px" class="center"/> <img src="images/gif/result6.gif" width="160px" class="center"/></center>

<center><img src="images/gif/presult1.gif" width="160px" class="center"/> <img src="images/gif/presult3.gif" width="160px" class="center"/> <img src="images/gif/presult5.gif" width="160px" class="center"/> <img src="images/gif/presult6.gif" width="160px" class="center"/> <img src="images/gif/presult7.gif" width="160px" class="center"/></center>

## Getting started
- Clone this repo
```
git clone https://github.com/sunshower76/Deep_Translation_Prior
cd Deep_Translation_Prior
```

- Start optimizing
```
bash optimizer.sh
```

We borrow code from public projects (huge thanks to all the projects). We mainly borrow code from  [CUT](https://github.com/taesungp/contrastive-unpaired-translation)
### BibTeX
If you find this research useful, please consider citing:
````BibTeX
@article{kim2021deep,
  title={Deep Translation Prior: Test-time Training for Photorealistic Style Transfer},
  author={Kim, Sunwoo and Kim, Soohyun and Kim, Seungryong},
  journal={arXiv preprint arXiv:2112.06150},
  year={2021}
}
````







