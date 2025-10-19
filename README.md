---
---

# ML-Project-Samuele-Magatti



## Project Objectives 

The main objective of this project is to implement a classification task using three different Convolutional Neural Networks (CNN). 

The expected output is supposed to be able to correctly classify the images of the Rock-Paper-Scissors dataset, available on Kaggle: [](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) 

into the labels rock, paper or scissors.  

 

## Dataset

Due to its size the dataset is not included in this repository.

It can be downloaded, as previously said, from kaggle : 


[Rock-Paper-Scissors Dataset on Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)



## Second dataset

External validation was performed on a private dataset that used different backgrounds. Due to privacy, it is not included.


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

In order to be able to run this project make sure you have installed Python 3.x

Install also the following libraries:



## Methodology


#### EDA
- Check dataset integrity: no duplicates or corrupted images, all images RGB, consistent size/aspect ratio
- Compute RGB means and distributions
- Visualizations

#### Preprocessing
- Split dataset into train/validation/test
- Compute mean and std only on training data
- Normalize using training values
- Data augmentation

#### CNN Construction & Training
- Implement 3 different CNNs
- Third CNN: grid search with K-fold CV for hyperparameter tuning

#### Model Evaluation
- Accuracy
- Train/validation loss curves
- Train/validation accuracy curves
- Confusion matrix
- F1 score


## External check

With the aim of checking the external validity of the results of the CNNs 
they has been tested using an homemade dataset 

## Results

All the analysis and the results are contained in the file report.pdf, for 
further informations about the CNNs results refer to the uppersaid file.


##  Notes

The files in the project structure are organized in the following way, 
the document called Samuele.Magatti.ML.Project.py contains the entire code, all 
the analysis has been don eusing this document, running time heavily depends on 
the device that you are going to use to execute the code, it can reaach 4 hours 
in the worst case scenario. Thus to guarantee reproducibility in a reasonable 
amount of time it has been added the file ml.project.faster.version.py that 
reducing the number of epochs and resizing all the images to 128x128 ( from 200
x300) can be much faster than the previuos, in running this document take into 
account a loss in accuracy. Moreover it must be said that for Linux and Windows
users there shall not be such a problem since the code will automatically set 
num_workers = 4 thus reducing the running time. 
In case you want to replicate just a part of the code you can go in the project
folder, there you shall find 7 subfolders, data.py contains all the data
prepocessing steps, the utilty.py contains the definitions of all the functions 
used in other files ( like set_seed) and the explorative data analisys, model.py
contains the definitions of all the three CNNs, first.cnn.py, second.cnn.py and 
third.cnn.py contains the code to execute the respective CNN finally the 
external.check.py contains the code necessary to check the results of the three 
CNNs using the homemade dataset.
For privacy reason the homemade dataset has not been added to this project, 
thus making the relative part the only non replicable.
The latter part of the code, containing the necessary parts to check the 
results of the three CNNs using the homemade dataset has not been added to the 
Samuele.Magatti.ML.Project.py document neither to the 
ml.project.faster.version.py to guarantee reproducibility of the two latter 
documents.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
