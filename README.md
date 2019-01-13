# Various-Machine-Learning-Assignments
These are assignments from my graduate Machine Learning (ML) course. They demonstrate the utility of various regression and classification techniques using real-world data.

## Repository Contents
This repository contains 5 real world data sets in .csv form. The rows represent unique samples and the columns represent unique features of each sample. The first row contains the label of each feature.

Input Data Set | Description                 | Example Features
---------------|-----------------------------| ---------------------------------------------------------------
cars.csv       |9 features of 392 car models | mpg, numCylinders, country of origin, etc 
movies.csv     |28 features of 2952 movies   | average rating, category, budget, etc
shaq.csv       |27 features of 1633 freethrows from Shaquille O'Neal | time, home game, losing, etc.
votes.csv      |26 features of 3113 U.S. counties across all 50 states during the 2016 presidential race | num Clinton votes, num Trump votes, percentage white/black/hispanic, ave income, etc.
wines.csv      |59 features of 28840 wines   | country of origin, wine type, price, review keywords, etc.

ML File                               | Description 
--------------------------------------|----------------------------------------------------------------------
AnalyzeWineReviews.R                  | These are words
ClassifyImagesKerasTensorFlowCNN.R    | These are words.
ISTA.R                                | These are words.
LogisticLassoRegression.R             | These are words.
PredictCarMPG_LassoRegression.R       | These are words.
PredictMovieRatings_adaBoost.R        | These are words.
PredictMovieRatings_RidgeRegression.R | These are words.
PredictShaqFreethrow_LogisticRidgeRegression.R | These are words.
PredictVotes_KerasTensorFlowNN.R      | These are words.
PredictVotes_KernelSVM.R              | These are words.
PredictVotes_KerasTensorFlowNN.R      | These are words.
PredictWinePrice_KerasTensorFlowNN.R  | These are words.


### Prerequisites
All of this software was written in the R programming language (version 3.5.1 (2018-07-02) -- "Feather Spray").
A few of the files utilize Tensorflow as well as the R Keras interface to Tensorflow.

### Installing
To install Tensorflow and R Keras, follow these instructions.
1. Install the package 'keras'
2. Load keras with library(keras) and then run install_keras(). (This takes a long time)

After doing this once, you may use the Tensorflow library by calling library(keras). Keras is a simplified interface to Tensorflow; it is a front-end for Tensorflow that makes Tensorflow more user-friendly.

## Running the tests


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


