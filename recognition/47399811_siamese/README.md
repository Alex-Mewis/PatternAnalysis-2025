# Siamese Network to Classify Melanoma on Lesions (ISIC 2020)
## Project Description
This project implements a Siamese model on the _ISIC 2020 Challenge dataset_  which aims to classify lesions as either benign or malignant. This project aims to achieve an accuracy of around 80% on the test dataset. This project will help dermatologists, who currently have to look through the lesions one by one, find early signs of skin cancer.


## Dataset Description
The dataset is hosted on Kaggle [here](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/overview). The dataset contains the following components:
- 33,126 images:
  * dimension: 1024x1024 
  * 32,542 images are benign (98.2% of all images)
  * 584 images are malignant (1.8% of all images)
- CSV file:
  * headers: ```image_name```, ```patient_id```, ```sex```, ```age_approx```, ```anatom_site_general_challenge```, ```diagnosis```, ```benign_malignant```, ```target```.
  * all of the meta data will be ignored in this project as we are only interested in classifying based on the images.
  * the ```target``` column is what we wish to predict
    * ```target == 0``` means the lesion is benign (harmless).
    * ```target == 1``` means the lesion is malignant (harmful).

Due to computation limits and redundancy in a high resolution 1024x1024 image the model was trained on a resized version of the dataset so all images are only 256x256. This resized dataset can be found [here](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized). Here are some sample images from the resized dataset across both classes:
<div style="text-align: center;">
<img src="_readme_figures/base_images_showcase.png" alt="[Base Image Plot]" width="45%"/>
</div>

## Enviroment Setup
Ensure you are in the ```47399811_siamese``` folder in the repository. If you are in the main directory of the repository, ```PatternAnalysis-2025``` run the following command:
```
cd recognition/47399811_siamese
```
to navigate to the correct directory.

### Adding the Dataset
Run the following commands to setup the directory for the data
```
mkdir data
mkdir data/images
```
Then download the dataset from [here](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized) and all of the images in the ```data/images``` folder and add the CSV file named ```ISIC_2020_Training_GroundTruth.csv``` to the ```data``` folder.

### Pyhon Setup
Download ```Python 3.13.7``` from [here](https://www.python.org/downloads/). Then to install of the required packages run the following command:
```
py -m pip install -r requirements.txt
```
Which will install all of the following packages:
- ```catppuccin == 2.5.0```
- ```grad-cam == 1.5.5```
- ```matplotlib == 3.10.6```
- ```numpy == 2.2.6```
- ```pandas == 2.3.3```
- ```pillow == 11.0.0```
- ```scikit-learn == 1.7.2```
- ```scipy == 1.16.2```
- ```seaborn == 0.13.2```
- ```torch == 2.8.0+cu126```
- ```torchvision == 0.23.0+cu126```

## Data Preporation
_All of the data loading, splitting, oversampling, batching and augumentation are performaned in ```dataset.py``` in the ```ISICImageDataset``` class._   

The images in ```data/images``` are split based into each class as given in ```data/ISIC_2020_TrainingGroundTruth.csv``` into two lists; the malignant images and benign images. Then we split the dataset into 3 groups:
- __Train Set__: 
  * 75% of malignant images and 75% of benign images.
  * Used to train the model.
- __Validation Set__:
  * 15% of malignant images and 15% of benign images.
  * Used to validate the performance of the model and tune hyperparameters.
- __Test Set__:
  * 15% of malignant images and 15% of benign images.
  * Used to evaluate the models final performance and tuning the hyperparameters. 

The original dataset has a large class imbalance with very little malignant images compared to benign images (1.8% to 98.2%). So a major focus on the data preparations went towards counter acting this imbalance because otherwise our model would end up not predicting any malignant cases which means the model would be useless. To counter act the class imbalance the minority class (malignant) was oversampled in the training data so that there was 50% of each class. Further more when iterating over the dataset in training the images would be batched with 50% of each class. 

To reduce over fitting by increasing the diversity in the dataset in the model multiple augmentations were applied to the images when training. The augmentations used in training are as follows:
-  ```RandomRotation(10)```: Rotates the image $\theta$ degrees where $\theta \in [0, 10]$. 
- ```RamdomVertialFlip()```: Flips the image vertical with a 50% probability.
- ```RandomHorizontalFlip()```: Flips the image horizontally with a 50% probability.
- ```ColorJitter(...)```: adjusts the brightness, contrast, saturation and hue of the image.

Additionally the pixel values of the images in training, validation and testing where normalised using ```Normalize(...)```. Here are some example sample images after the train transformations have been applied:
<div style="text-align: center;">
<img src="_readme_figures/transformed_images_showcase.png" alt="[Transformed Image Plot]" width="45%"/>
</div>


## Model Architecture
The model used for classification has been split into two separate networks; the siamese network and a binary classifier.

_Both networks have beeen implemented using ```Pytorch``` in ```modules.py```._ 
### Networks
The siamese network ```SiameseNetwork``` consists of a convolution neural network (CNN) backbone and a simple binary classifier. The CNN backbone takes in the images and produces some latent vector output. While the binary classifier uses the backbones latent vector output to classify the image. The latent vector is some compact representation of the image which keeps important features of the image which will be used to distinguish images of separate classes.

For this project a slightly modified ```restnet18``` was used for the CNN backbone of the siamese network. The ```resnet18```'s architecture is shown below:

<div style="text-align: center;">
<img src="_readme_figures/Original-ResNet-18-Architecture.png" alt="[Base Image Plot]" width="80%"/>

<em>Original ResNet-18 Architecture (Farheen et al., 2019)</em>
</div>


For this project the last fully connected layer and softmax have been removed, as we are only need the CNN part of the model. As can be seen above this model will output a 512 dimensional latent vector representation of the image. This latent vector will then be inputted into the binary classifier.

The binary classifier has been implemented using fulling connected layers. The model consists of 4 fully connected layers of size: $512 \to 256 \to 128 \to 64 \to 2$ with a ```RelU``` activation function and ```Dropout(0.5)``` between each fully connected layer. Notice the input dimension is 512 which exactly the dimension of the output latent vector from the siamese network. The final layer has two values; the first being the models confidence that the image is benign (```label==0```) and the second is the models confidence the image is malignant (```label==1```). The models confidence in each label are used to generate its predictions. The dropout layers in the classifier help to reduce over fitting in the model by randomly turning off 50% of the nodes in each layer.

### Loss Functions
The goal of the CNN backbone is to make the latent vectors of two images of the same label as close together while the latent vectors of two image with different labels far apart. To achieve this the ```TripletMarginLoss``` was used to evaluate the networks performance. The triplet loss works by for each anchor image $\mathcal{A}$ in a batch we randomly choose a 2 other images $\mathcal{P}$ and $\mathcal{N}$ such that the label of $\mathcal{A}$ and $\mathcal{P}$ are the same while the label of $\mathcal{N}$ differs. Then we reward the latent vector repersentation of $\mathcal{A}$ and $\mathcal{P}$ being close together and $\mathcal{A}$ and $\mathcal{N}$ beign far apart. Formally the triplet loss function on a batch of size $N$ can be defined as follows,
$$L(\bm{a}, \bm{p}, \bm{n}) := \max_{i\in\{1,\ldots,N\}} \{\underbrace{\|\bm{a}_i - \bm{p}_i \|}_{\text{distance between two images of the same label}} - \underbrace{\|\bm{a}_i - \bm{n}_i \|}_{\text{distance between two images of different labels}} + \text{margin}, 0\}.$$
where: 
- $\bm{a} := (\bm{a}_1, \ldots, \bm{a}_N)$ where $\bm{a}_i$ is the latent vector output of the siamese network on the $i^{\text{th}}$ anchor image in the batch.
- $\bm{p} := (\bm{p}_1, \ldots, \bm{p}_N)$ where $\bm{p}_i$ is the latent vector output of the siamese network on a chosen _positive_ image which has the same label as $\bm{a}_i$.
- $\bm{n} := (\bm{n}_1, \ldots, \bm{n}_N)$ where $\bm{n}_i$ is the latent vector output of the siamese network on a chosen _negative_ image which has a different label than $\bm{a}_i$.

For this project we have also chosen $\text{margin} = 1$ (which is a common value).


The goal of the binary classifier is to correctly classify the images from their latent vectors as either benign or malignant. So to achieve this we use the ```CrossEntropyLoss``` which measures how close the models probability distribution for the labels is to the true labels. To define the Cross Entropy Loss we must first define the models probability distribution from its confidence output in the final layer of the classifier. This can be achieved using the $\text{softmax}$ function, that is, if we let $(c_0, c_1)$ be the classifiers output (the models confidence in benign and malignant respectfully) then we can define the classifiers probability of the image being benign $p_0$ or malignant $p_1$ as follows:
$$p_0:=\text{softmax}(c_0) = \frac{e^{c_0}}{e^{c_0}+e^{c_1}}, \hspace{1cm} p_1 := \text{softmax}(c_1) = \frac{e^{c_1}}{e^{c_0} + e^{c_1}}.$$
Then using these probabilities we define Cross Entropy Loss ($\text{CE}$) on a batch of size $N$ for true labels $\bm{y} = (y_1, \ldots, y_N)$ as,
$$CE = -\frac{1}{N}\sum_{i=1}^N \left[y_i\cdot \log(p_{i,0}) +  (1-y_i)\log(p_{i,1}) \right],$$
where $p_{i,\ell}$ is the models probability of the $i^{\text{th}}$ images label being $\ell$.

Finally the loss of the model is just the sum of the ```TripletMarginLoss``` to evaluate the backbone and ```CrossEntropyLoss``` to evaluate the binary classifier. We want both networks to train simultaneously and equally so they have to be combined into a single loss function.

## Training
_The model was trained using the functions in ```train.py```. While some of the models hyperparameters are given in ```config.py```._
The following hyperparameters have been used to training the model:
- ```batch_size=32```
- ```epochs=16```
- ```learning_rate=0.01```, a learning rate scheduler, ```CosineAnnealingLR```, was used with ```T_max=25, eta_min=1e-7```


### Training Results
The models loss, accuracy and AUC-ROC score during training has been plotted for each epoch as shown below:
<div style="text-align: center;">
<img src="_readme_figures/model_1761827625.979611.png" alt="[]" width="100%"/>
</div>

**Analysis**:
- Validation loss is closely following the training loss which means the model is correctly generalising and is unlikely to be over fit.
- Near the end of training the models training loss continues to drop rapidly while the validation loss stabilises. This show the training was terminated at the right time before the model would start to over fit.
- The models training and validation accuracy closely followed each other meaning there was good generalisations in the models predictions.
- The models training AUC-ROC value was rapidly increasing towards 1 while the validation score did not increase as much. This shows the model was able to build a lot more confidence in its predictions on the training set when compared to the validation set.

The t-SNE plot on the validation set is shown below:
<div style="text-align: center;">
<img src="_readme_figures/tsne_scatter_validation_1761765127.22697.png" alt="[Base Image Plot]" width="50%"/>
</div>

**Analysis**:
- Some separation can be seen between the two classes, as a good amount of malignant images are pooled together at the end of the shape (to the right). While the benign images are mostly all grouped together in the large blob on the left end.
- While there is still a fair bit of overlap between the two classes this comes as a side effect of projecting the 512 dimensional latent vectors into a 2D plane. 

## Evaluation
To evaluate the reasonableness of the CNN backbone it is useful to know what the model is looking at in these images to extract its features (the latent vectors). Because it could be _cheating_ by using some unintended artifact in the images not be correctly highlighting the lesions in the image. To check the models reasonableness we can look at the output of GRAD-CAM's on a random sample of images from the dataset:
<div style="text-align: center;">
<img src="_readme_figures/grad_cam_image_showcase.png" alt="[Base Image Plot]" width="45%"/>
</div>

**Analysis**:
- The ares in the image where the overlay is brightest are where the model is looking at the most when extracting its features.
- The model, for the most part, is seen to be looking at the lesions. This is good as this should be what is used to distinguish one image as either benign or malignant.
- While sometimes model is not able to highlight the lesions this is potentially a sign the model was over fit or insufficiently trained to perfectly pick up every mole. Either the model is not _cheating_ as it is not highlighting some obscure part of every single image.


## Improvements and Future Directions

## References
https://www.researchgate.net/figure/Original-ResNet-18-Architecture_fig1_336642248

