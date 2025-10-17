---
---

# ML-Project-Samuele-Magatti



## Project Objectives 

The main objective of this project is to implement a classification task using three different Convolutional Neural Networks (CNN). 

The expected output is supposed to be able to correctly classify the images of the Rock-Paper-Scissors dataset, available on Kaggle: [](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) 

into the labels rock, paper or scissors.  

 

## Dataset

Due to its size the dataset is not included in this repository.

It can be downloaded, as previously said, from kaggle [](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)



[Rock-Paper-Scissors Dataset on Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)



## Second dataset

A second homemade dataset with different backgrounds has been used as an external check for the predictors, for privacy reason it was not Made public.



## Project Structure

ML-Project-Samuele-Magatti/

├── data/

├── Samuele.Magatti.ML.Project.py

├── report.pdf

├── requirements.txt

├── ml.project.faster.version.py

├──.gitignore

├── Project

           └── data.py

           └── first.cnn.py

           └── second.cnn.py

           └── third.cnn.py

           └── model.py

           └── utility.py

           └── final.check.py

└── README.md



## Installation and requirements

In order to be able to run this project make sure you have installed Phyton 3.x

Install also the following libraries:



## Methodology



-##Eda :



check the data are installed correctly, that there are no duplicstes nor corrupted images, that all the images are rgb and the same size and aspect ratio, 
check the rgb means and distributions
graphs 


- ##Prepocessing of data : 



split the data in train validation and test
compite the mean and standard deviati in only for the train part
normalization using the previuosly computed values
Data augmentation 


- ## Construction and training of three different CNNs

in the last one it has bene used best ed cross validation to choose the best hyperparameters 


-## Models evaluation ( for each CNN):



Accuracy, 
train and validation loss graphs
Train and validation accuracy graphs
Lable reshuffle 
Confusion Matrix 


## External check
using an homemade dataset there is a new check for 
## Results



## License and Note

