
# In2I : Unsupervised Multi-Image-to-Image Translation Using Generative Adversarial Networks

This code is the implementation of the paper, <i> In2I : Unsupervised Multi-Image-to-Image Translation Using Generative Adversarial Networks</i>. Implementation is based on the CycleGAN PyTorch code written by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung89). Implementation for the traslation of MRI to CT with two or three input modalities.



#### In2I : [[Project]](https://github.com/PramuPerera/In2I) [[Paper]](https://arxiv.org/abs/1711.09334)


### 
2 input modalities: place MRI data in 2 folders: 'trainA_in' and 'trainA_out' for train, 'testA_in' and 'testA_out' for test

3 input modalities: place MRI data in 3 folders: 'trainA_inT2', 'trainA_outT2' and 'trainA_t2' for train, 'testA_inT2', 'testA_outT2' and 'testA_t2' for test

CT data in: 'trainB' and 'testB' folders

different modalities of the same slice must have the same name and be placed in the proper folder

- Train a model 2 input:
```
!python train.py --dataroot ./datasets/MRI2CT_inOutJoin --name MRI2CT_inOutJoin --model cycle_gan --no_dropout --no_input 2 --input_nc 1 --input_nc2 1 --output_nc 3 --display_id 0 --niter 150 --niter_decay 50 
```
- Train a model 3 input:
```
!python train.py --dataroot ./datasets/MRI2CT_inOutJoin --name MRI2CT_t2Join --model cycle_gan --no_dropout --no_input 3 --input_nc 1 --input_nc2 1 --output_nc 3 --display_id 0 --gpu_ids 1 --niter 150 --niter_decay 50 
```
- Test the model 2 input:
```
!python test.py --dataroot ./datasets/MRI2CT_inOutJoin/ --name MRI2CT_inOutJoin --phase test --no_input 2 --no_dropout --input_nc 1 --input_nc2 1 --output_nc 3 --how_many 1294 --preprocess resize --loadSize 256
```
- Test the model 3 input:
```
!python test.py --dataroot ./datasets/MRI2CT_inOutJoin/ --name MRI2CT_t2Join --phase test --no_input 3 --no_dropout --input_nc 1 --input_nc2 1 --output_nc 3 --how_many 1294 --preprocess resize --loadSize 256
```

The test results will be saved in the 'results' folder

It should be noted that all arguments  are similar to that of CycleGAN PyTorch code. input_nc and input_nc2 specifies number of channels in each modality Generator architechture is based on ResNET and is fixed.   

