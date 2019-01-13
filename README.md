# Various-Machine-Learning-Assignments
These are assignments from my graduate Machine Learning (ML) course. They demonstrate the utility of various regression and classification techniques using real-world data.

## Repository Contents
This repository contains 5 real world data sets in .csv form. The rows represent unique samples and the columns represent unique features of each sample. The first row contains the labels of each feature.

Input Data Set | Description                 | Example Features
---------------|-----------------------------| ---------------------------------------------------------------
cars.csv       |9 features of 392 car models | mpg, numCylinders, country of origin, etc 
movies.csv     |28 features of 2952 movies   | average rating, category, budget, etc
shaq.csv       |27 features of 1633 freethrows from Shaquille O'Neal | time, home game, losing, etc.
votes.csv      |26 features of 3113 U.S. counties across all 50 states during the 2016 presidential race | num Clinton/Trump votes, percentage white/black/hispanic, ave salary, education levels, etc.
wines.csv      |59 features of 28840 wines   | country of origin, wine type, price, review keywords, etc.

This repository contains 11 executable .R files which apply ML algorithms to contruct predictive models based on the data sets described above, as well as synthetic randomly generated features/models/responses. 
Generally speaking, these predictive models perform either *regression* (i.e. estimation of continuous values) or classification given input feature sets (i.e. predictors).
The supervised predictive models include linear regression (e.g. OLS), logistic regression, regularized regression (Ridge and Lasso), support vector machines (SVM), ensemble methods (adaBoost, gradientBoost), K nearest neighbours, and neural networks (including CNN).
The methodology for constructing supervised predictive models revolves around decomposing data sets in training, validation, and test groups.
Moreover, particular emphasis is placed on *Cross Validation* for producing optimal model parameters.
Evaluation of regression and classification models are based on mean squared error and mean misclassification rate, respectively. 
Additionally, unsupervised learning methods provided here include Principal Component Analysis (PCA), and K-means clustering.

ML File                               | Description 
--------------------------------------|----------------------------------------------------------------------
AnalyzeWineReviews.R                  | Apply PCA and K-means to 29,000 wine reviews. (Pinot Noir, Chardonnay, and Riesling)
ClassifyImages_KerasTensorFlowCNN.R   | Apply Keras/Tensorflow Convolutional Neural Networks to classify the type of clothing based on an image of the clothing (i.e. *dataset_fashion_mnist()*).
ISTA.R                                | Support file for Lasso Regularization. Implements the *Iterative Soft Thresholding Algorithm*.
LogisticLassoRegression.R             | Performs Logistic Lasso on randomly generated model coefficients, input features, and output labels (i.e. synthetic data for demonstration purpose).
NonNegativeOLS.R                      | Demonstrates how to apply OLS while 'projecting' the values to be non-negative due to a-priori knowledge of that the parameters are non-negative.
PredictCarMPG_LassoRegression.R       | Apply refitted lasso regression to predict the miles/gallon of cars. (Uses ISTA.R)
PredictMovieRatings_adaBoost.R        | Apply adaBoost (decision stump weak learners) to classify movies as either above or below average rated. (adaBoost is only for classification)
PredictMovieRatings_RidgeRegression.R | Apply Ridge Regression with Cross Validation to predict movie ratings.
PredictShaqFreethrow_LogisticRidgeRegression.R | Apply Logistic Ridge Regression to predict whether or not Shaq will make a free throw. Highly unpredictable!
PredictVotes_KerasTensorFlowNN.R      | Apply Keras/Tensorflow Neural Networks to predict whether a county voted for Trump of Clinton.
PredictVotes_KernelSVM.R              | Apply Kernel SVM with Polynomial Kernel to predict whether a county voted for Trump of Clinton.
PredictWinePrice_KerasTensorFlowNN.R  | Apply Keras/Tensorflow Neural Networks to predict the price a wine sold for based on select words from the corresponding wine review.

Due to the instructive nature of the assignments, predictive models here were mostly developed from scratch (even though R provides built in functionality).

### Prerequisites
All of this software was written in the R programming language (version 3.5.1 (2018-07-02) -- "Feather Spray").
A few of the files utilize Tensorflow as well as the R Keras interface to Tensorflow.

### Installing Tensorflow and R Keras
1. Install the package 'keras'
2. Load keras with library(keras) and then run install_keras(). (This takes a long time)

After doing this once, you may use the Tensorflow library by calling library(keras). Keras is a simplified interface to Tensorflow; it is a front-end for Tensorflow that makes Tensorflow more user-friendly.

## Running the tests
Make sure that the input data .csv files are in the same directory as the .R files.
Navigate via commandline to the root directory of this repository.
Enter the R environment by typing
> R

Next, to run a file, type
> Rscript NameOfFile.R

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

> We're living the future so
> the present is our past.

Kanye West
