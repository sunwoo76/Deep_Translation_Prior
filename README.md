# Official implementation of DTP(AAAI'22)

This is the official implementation of our paper :

**Deep Translation Prior: Test-time Training for Photorealistic Style Transfer(AAAI 2022)**

Authors: Sunwoo Kim, Soohyun Kim and [Seungryong Kim](https://seungryong.github.io/)

You can check out the paper on [[arXiv](https://arxiv.org/abs/2112.06150)].

# Network

Our model DTP is illustrated below:

![alt text](/images/network.png)

## Example Results
<p align="center">
<img src="images/gif/result_1.gif" width="400px"/> <img src="images/gif/result_2.gif" width="400px"/>
 </p>
<!-- <p align="center">
<img src="images/content.png" width="130px"/>  <img src="images/content/result1_c.png" width="130px"/> <img src="images/content/result2_c.png" width="130px"/> <img src="images/content/result4_c.png" width="130px"/> <img src="images/content/result5_c.png" width="130px"/> <img src="images/content/result6_c.png" width="130px"/>
</p>
<p align="center">
 <img src="images/style.png" width="130px"/><img src="images/style/result1_s.png" width="130px"/> <img src="images/style/result2_s.png" width="130px"/> <img src="images/style/result4_s.png" width="130px"/> <img src="images/style/result5_s.png" width="130px"/> <img src="images/style/result6_s.png" width="130px"/>
</p>
<p align="center">
<img src="images/result.png" width="130px"/> <img src="images/gif/result1.gif" width="130px"/> <img src="images/gif/result2.gif" width="130px"/> <img src="images/gif/result4.gif" width="130px"/> <img src="images/gif/result5.gif" width="130px"/> <img src="images/gif/result6.gif" width="130px"/>
</p>



<p align="center">
<img src="images/content.png" width="130px"/> <img src="images/content/presult1_c.png" width="130px"/> <img src="images/content/presult3_c.png" width="130px"/> <img src="images/content/presult5_c.png" width="130px"/> <img src="images/content/presult6_c.png" width="130px"/> <img src="images/content/presult7_c.png" width="130px"/>
</p>
<p align="center">
<img src="images/style.png" width="130px"/> <img src="images/style/presult1_s.png" width="130px"/> <img src="images/style/presult3_s.png" width="130px"/> <img src="images/style/presult5_s.png" width="130px"/> <img src="images/style/presult6_s.png" width="130px"/> <img src="images/style/presult7_s.png" width="130px"/>
</p>
<p align="center">  <img src="images/result.png" width="130px"/> <img src="images/gif/presult1.gif" width="130px"/> <img src="images/gif/presult3.gif" width="130px"/> <img src="images/gif/presult5.gif" width="130px"/> <img src="images/gif/presult6.gif" width="130px"/> <img src="images/gif/presult7.gif" width="130px"/>
</p> -->


## Getting started
- Clone this repo
```
git clone https://github.com/sunwoo76/Deep_Translation_Prior
cd Deep_Translation_Prior
```

- Start optimizing
```
bash optimizer.sh
```

## Acknowledgement
We borrow codes and datasets from public projects. We mainly borrow code from  [CUT](https://github.com/taesungp/contrastive-unpaired-translation)

You can download the datasets used in this paper from the github repositories of [WCT2](https://github.com/clovaai/WCT2), [StyleNas](https://github.com/pkuanjie/StyleNAS),  [FFHQ](https://github.com/NVlabs/ffhq-dataset) and [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans)

## Citation
If you find this research useful, please consider citing:
````BibTeX
@article{kim2021deep,
  title={Deep Translation Prior: Test-time Training for Photorealistic Style Transfer},
  author={Kim, Sunwoo and Kim, Soohyun and Kim, Seungryong},
  journal={arXiv preprint arXiv:2112.06150},
  year={2021}
}
````







