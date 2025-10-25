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
  * all of the meta data will be ignored in this project as we are only interested in classifing based on the images.
  * the ```target``` column is what we wish to predict
    * ```target == 0``` means the lesion is benign (harmless).
    * ```target == 1``` means the lesion is malignant (harmful).

Due to computation limits and redundancy in a high resolution 1024x1024 image the model was trained on a resized version of the dataset so all images are only 256x256. This resized dataset can be found [here](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized). There is a major class imbalance in the dataset as very little of the images are malignant. So a heavy focus for this project was put to counter acting this imbalance by oversampling the minory class and applying augmentations to the images.

## Enviroment Setup