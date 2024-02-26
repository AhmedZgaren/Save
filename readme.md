# SAVE: Self-Attention mechanism for zero-shot generic object counting via Visual Embedding from objects instance features
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
```
## Download trained model
We provide pre-trained model weights in FSC147:

[model](https://drive.google.com/file/d/1BsLVBWhxFkum-JdHSCmoEL39m4IvOzsO/view?usp=drive_link)

Store the pre-trained model in the pre-trained folder. 
## Download pretrained backbone on FSC147
[backbone]()

Store the backbone weigths in the pretrained folder.
## Exemple test
To test the model in an image use the code below:
```bash
python test.py --path\to\img
```
## Train the model from scratch

To train the model from scratch, run the code bellow:
### Download FSC147 dataset
Use this [link]() to download the FSC147 dataset and store images under the folder `.\data\images\`  
### Generate the synthetic data
run the script `augment.py` to generate synthetic data from FSC147 images
```bash
python augment.py
```
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
