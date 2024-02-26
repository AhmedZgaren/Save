# SAVE: zero-shot generic object counting with Self-Attention on Visual Embedding
>Ahmed Zgaren<sup>1,2</sup>, Wassim Bouachir<sup>2</sup>, Nizar Bouguila<sup>1</sup>

>><sup>1</sup> CIIS, Concordia University,

>><sup>2</sup> Data science Lab, TELUQ University

![img](overview.jpg)
## Environment 
Here is the code to setup the conda environment:
```bash
git clone https://github.com/AhmedZgaren/Countvit.git
cd Countvit
conda env create -f environment.yml
conda activate count
```
## Download trained model
We provide pre-trained model weights in FSC147:

[model](https://drive.google.com/file/d/1wvARqtm7dA28f5Hs5zlETF5OlkeT4osY/view?usp=drive_link)

Store the pre-trained model in the pre-trained folder. 
## Download pretrained backbone on FSC147
[backbone](https://drive.google.com/file/d/1xnouES1ZJ0qJoubpWjNCHmbNtc6Jn2lu/view?usp=sharing)

Store the backbone weigths in the pretrained folder.
## Demo
To test the model in an image use the code below:
```bash
python demo.py --i path\to\img
```
## Train the model from scratch

To train the model from scratch, run the code bellow:
### Download FSC147 dataset
Use this [link](https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing) to download the FSC147 dataset and store images under the folder `.\data\images\`  
<!-- ### Generate the synthetic data
run the script `augment_mix.py` to generate synthetic data from FSC147 images
```bash
python augment_mix.py
```
Use this [link](https://drive.google.com/file/d/1JAt5w5GrXn2V_rndw0RVHszttHUHofBP/view?usp=sharing) to download the bboxes annotations and store it under `.\data\labels` -->
### Train your model
To train the model just run the following command:
```bash
python train.py
```
### Test the trained model

To test the model in the FSC147 validation set:
```bash
python valid.py --valid
```

To test the model in the FSC147 test set:
```bash
python valid.py --tst
```
