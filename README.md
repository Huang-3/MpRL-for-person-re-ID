# MpRL-for-person-re-ID
This repository contains the code for our paper [Multi-Pseudo Regularized Label for Generated Data in Person Re-Identification](https://ieeexplore.ieee.org/abstract/document/8485730).

The code is mainly modified from [Person-reID_GAN](https://github.com/layumi/Person-reID_GAN).

### To run this code

### 1.Data Generation (GAN)
The first stage is to generate fake images by DCGAN.
We use the DCGAN code at https://github.com/layumi/DCGAN-tensorflow.
You can also directly download our generated data (24000 generated images for Market1501) from [Google_Drive](https://drive.google.com/open?id=1-Qv8QfmLi24svclJ3Ee-6y5Zk6HLjZfP)

### 2.Semi-supervised Learning
This repository includes two baseline code and the dMpRL-II method in our paper.

| Models               | Reference | 
| --------              | -----  | 
| train_res_iden_baseline.m        | ResNet50 baseline (only use real data) | 
| train_res_iden_MpRL3.m    | assign dMpRL-II label for generated images (combine real and generated data)|  
| train_2stream_baseline.m    | ResNet50 two-stream baseline (only use real data) | 
| train_2stream_MpRL3.m | assign dMpRL-II label for generated images (combine real and generated data)| 

* We propose MpRL virtual labels for generated data. Three strategies are used to train the combination of real and generated data. We named the three strategies as sMpRL, dMpRL-I and dMpRL-II respectively in our paper. You can find more detailed code for our MpRL Loss in:

sMpRL (the training code will be added to this repos soon):
https://github.com/Huang-3/MpRL-for-person-re-ID/blob/master/matlab/%2Bdagnn/Pseudo_Loss_Multi_Static.m

dMpRL-I (the training code will be added to this repos soon):
https://github.com/Huang-3/MpRL-for-person-re-ID/blob/master/matlab/%2Bdagnn/Pseudo_Loss_Multi_Dynamic_1.m

dMpRL-II (Our best method, used in this repos):
https://github.com/Huang-3/MpRL-for-person-re-ID/blob/master/matlab/%2Bdagnn/Pseudo_Loss_Multi_Dynamic_2.m


### Compile Matconvnet
We use the Matconvnet package to implement our code, you can juse download this repos and run `gpu_compile.m` in Matlab to compile functions in this repos. We have included the Matconvnet in this repos.
(We use Cuda-8.0 and Cudnn-5.1, our code do not support cudnn version > 5.1. If you have any problem in compiling, first, try to check your cudnn version.) 

### Dataset
We take Market1501 as an example in this repos.
Download [Market1501 Dataset](http://www.liangzheng.org/Project/project_reid.html)

### Training and Testing
1. Download the [ResNet-50 model](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat) pretrained on Imagenet. Put it in the `result` dir.

2. Prepare data:
   For real data: `code/prepare_data/prepare_data.m`.
   For real + generated data: `code/prepare_data/prepare_gan_data.m`

3. Run the training code:
   Simply run the  `train.m`, the trained model will be saved at `result/Model_Name/`
   
4. Evaluation:
   Run `test/test_gallery_query_crazy.m` to extract feature of images in the gallery and query set. They will store in a .mat file in `test`. Then you can use it to do evaluation.
   
   Run `evaluation/zzd_evaluation_res_faster.m`
   
### Citation
Please cite this paper in your publications if it helps your research:
```
@article{huang2019multi,
  title={Multi-Pseudo Regularized Label for Generated Data in Person Re-Identification},
  author={Huang, Yan and Xu, Jingsong and Wu, Qiang and Zheng, Zhedong and Zhang, Zhaoxiang and Zhang, Jian},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={3},
  pages={1391--1403},
  year={2019},
  publisher={IEEE}
}
