# FPHGD-Net
## Feature-domain Proximal High-dimensional Gradient Descent Network for Image Compressed Sensing (ICIP 2023)
This repository is for FPHGD-Net introduced in the following paper：

Liang Z, Yang C. Feature-Domain Proximal High-Dimensional Gradient Descent Network for Image Compressed Sensing[C]//2023 IEEE International Conference on Image Processing (ICIP). IEEE, 2023: 1475-1479.

## :art: Dataset

For training, we use 400 images from the training set and test set of the BSDS500 dataset. The training images are cropped to 100000 96*96 pixel sub-images with data augmentation. For testing, we utilize three widely-used benchmark datasets, including Set11, BSDS68 and Urban100. 

## 🔧 Requirements
- Python == 3.10.4
- Pytorch == 1.12.1

## :computer: Command
### Train
- For FPHGD-Net-Tiny:
`python train_code_tiny.py --cs_ratio 0.01/0.05/0.1/0.3/0.5`
- For FPHGD-Net:
`python train_code_middle.py --cs_ratio 0.01/0.05/0.1/0.3/0.5`
### Test
- For FPHGD-Net-Tiny:
`python test_code_tiny.py --cs_ratio 0.01/0.05/0.1/0.3/0.5 --test_name Set11/Urban100`
- For FPHGD-Net:
`python test_code_middle.py --cs_ratio 0.01/0.05/0.1/0.3/0.5 --test_name Set11/Urban100`
## :Citation
If you find the code helpful in your research or work, please cite our papers:      
```
@INPROCEEDINGS{10222347,
  author={Liang, Ziwen and Yang, Chunling},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)}, 
  title={Feature-Domain Proximal High-Dimensional Gradient Descent Network for Image Compressed Sensing}, 
  year={2023},
  volume={},
  number={},
  pages={1475-1479},
  keywords={Integrated circuits;Phase measurement;Image coding;Artificial neural networks;Sensors;Image restoration;Compressed sensing;image compressed sensing;deep learning;image restoration;inverse problem},
  doi={10.1109/ICIP49359.2023.10222347}}
```
