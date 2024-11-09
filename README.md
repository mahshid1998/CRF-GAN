# CRF-GAN

PyTorch implementation for paper *Memory-Efficient 3D High-Resolution Medical Image Synthesis Using CRF-Guided GANs* Accepted for Artificial Intelligence for Healthcare Applications, 3rd International Workshop, ICPR 2024.

### Requirements
```bash
conda env create --name crfgan -f environment.yml
conda activate crfgan
```

### Data Preprocessing
The volume data need to be cropped or resized to 128<sup>3</sup> or 256<sup>3</sup>, and intensity value need to be scaled to [-1,1]. In addition, we would like to advise you to trim blank axial slices. More details can be found at
```bash
python preprocess.py
```

### Training
##### for unconditional set num-class to 0
```bash
python train.py --workers 8 --img-size 256 --num-class N --exp-name 'crf-gan1' --data-dir DATA_DIR
```

### Testing
```bash
visualization.ipynb
python evaluation/fid_score.py
```
