# Finetuning Mask2Former with DiNAT backbone on custom Dataset

## Introduction
Mask2Former is a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic). Its key components include masked attention, which extracts localized features by constraining cross-attention within predicted mask regions. In addition to reducing the research effort by at least three times, it outperforms the best specialized architectures by a significant margin on four popular datasets. Most notably, Mask2Former sets a new state-of-the-art for panoptic segmentation (57.8 PQ on COCO), instance segmentation (50.1 AP on COCO) and semantic segmentation (57.7 mIoU on ADE20K).

### [Mask2Former Project page](https://github.com/facebookresearch/Mask2Former) | [Mask2Former Paper](https://arxiv.org/abs/2112.10764) | 
Run our demo using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uIWE5KbGFSjrxey2aRd5pWkKNY1_SaNq)
### [Mask2Former Hugging Face](https://huggingface.co/docs/transformers/model_doc/mask2former)

[DiNAT] ![intro_dark](https://github.com/sleepreap/Finetune-DiNAT/assets/98008874/3ed1040a-3d1e-4875-9baf-01bbfb8c6ce2)


Neighborhood Attention (NA, local attention) was introduced in original paper, 
[NAT](NAT.md), and runs efficiently with extension to PyTorch, [NATTEN](https://github.com/SHI-Labs/NATTEN).

DiNat is a new model, [DiNAT](DiNAT.md), which extends NA by dilating neighborhoods (DiNA, sparse global attention, a.k.a. dilated local attention).

Combinations of NA/DiNA are capable of preserving locality, maintaining translational equivariance, expanding the receptive field exponentially, 
and capturing longer-range inter-dependencies,  leading to significant performance boosts in downstream vision tasks.

###  [NAT Hugging Face](https://huggingface.co/models?filter=nat) 
### [DiNAT Hugging Face](https://huggingface.co/models?filter=dinat)

## Purpose
The purpose of this document is to build a process of finetuning Mask2Former using DiNAT backbone for custom dataset on semantic segmentation. The code is done using Pytorch Lightning and the model can be imported from hugging face.

1. Create a virtual environment: `conda create -n DiNAT python=3.8 -y` and `conda activate DiNAT `
2. Download code: `git clone https://github.com/sleepreap/Finetune-DiNAT.git`
3. Install [Pytorch CUDA 11.8](https://pytorch.org/): ` pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 `
4. `cd Finetune-DiNAT` and run `pip install -e .`
5. Install [Cmake]: `pip install cmake `
6. Install [NATTEN]: `pip3 install natten==0.17.1+torch230cu121 -f https://shi-labs.com/natten/wheels/`

## Dataset
Use createDataset.py to create the folders.
Refer to the README file in the folder "Data" on where to upload the images and labels.

## Training
1. 'cd scripts' 
2. set up the configs required in config.py
3. run the train.py file

A CSVlogger and the trained model file will be saved after the training has been completed. The model file would be saved as "Mask2Former.ckpt" in the same directory. An "output" folder will be created to store the contents of the CSVlogger.

## Testing
The testing is done using Mean-IOU, as well as pixel accuracy from the evaluate package. It will provide individual accuracy and IOU scores for each class label specified, as well as the mean accuracy and IOU scores of all the class labels. To run the test file, the model path of the trained model must be provided as an argument.

1. 'cd scripts' 
2. run the test.py file using this command: python test.py --model_path MODEL_PATH
   
```bash
e.g python test.py --model_path Mask2Former.ckpt
```

## Utilities
This folder contains the following scripts:
1. inference.py
2. saveComparison.py
3. predictionOverlay.py
4. saveComparisonWithOverlay.py
   
### All the scripts already reference the parent folder for the command line arguments. The images and labels used are from the test dataset respectively.

Inference.py would save all the predictions by the model on the test dataset in the specified save path folder.



```bash
1. 'cd scripts/utilities'
2. run the inference.py file using this command: python inference.py --model_path MODEL_PATH --save_path SAVE_PATH
```

saveComparison.py would save a plot of the prediction and the ground truth side by side in the specified save path folder. There is only 1 comparison per image due to memory constraint.

```bash
1. 'cd scripts/utilities'
2. run the saveComparison.py file using this command: python saveComparison.py --model_path MODEL_PATH --save_path SAVE_PATH
```
predictionOverlay.py would save the overlay that shows the TP+TN+FP+FN of the predictions done by the model for all the images in the specified save path folder. Black means TN (background), Green means TP (metal-line), Red means FN (metal-line as background), Blue means FP (background as metal-line).

```bash
1. 'cd scripts/utilities'
2. run the predictionOverlay.py file using this command: python predictionOverlay.py --model_path MODEL_PATH --save_path SAVE_PATH
```
saveComparisonWithOverlay.py would save a plot of the overlay and the ground truth side by side in the specified save path folder. There is only 1 comparison per image due to memory constraint.

```bash
1. 'cd scripts/utilities'
2. run the saveComparisonWithOverlay.py file using this command: python saveComparisonWithOverlay.py --model_path MODEL_PATH --save_path SAVE_PATH
```

## Citation
```BibTeX
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
@inproceedings{hassani2023neighborhood,
	title        = {Neighborhood Attention Transformer},
	author       = {Ali Hassani and Steven Walton and Jiachen Li and Shen Li and Humphrey Shi},
	booktitle    = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	month        = {June},
	year         = {2023},
	pages        = {6185-6194}
}
@article{hassani2022dilated,
	title        = {Dilated Neighborhood Attention Transformer},
	author       = {Ali Hassani and Humphrey Shi},
	year         = 2022,
	url          = {https://arxiv.org/abs/2209.15001},
	eprint       = {2209.15001},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}
```
