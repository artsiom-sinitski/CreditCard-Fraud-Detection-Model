
# CreditCard-Fraud-Detection-Model

Determines if a credit card was used maliciously, not by the original owner

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [Team](#team)
- [FAQ](#faq)
- [Support](#support)
- [License](#license)

## Installation

#### Install Python 3.7+
#### Install the following Python packages:
- Keras (DL high level framework)
- Tensorflow (DL backend)
- pandas (data manipulation)
- scikit-learn (model crossvalidation)
- matplotlib (data visualization)

## Clone

- Clone this repo to your local machine using `https://github.com/artsiom-sinitski/CreditCard-Fraud-Detection-Model.git`

## Features
Predicting fraudulent credit card activities is a challenge for many e-comm companies and banks.
The dataset provided under the 'data/' directory is a real data set annonimized for security reasons
and the one we will use to explore a number of different machine learning techniques:
* Deep Neural Network w/ Dense & Dropout layers
* Support Vector Machine (SVM) algorithm
* Random Forest & Decision Tree algorithms
* Confusion Matrix - to gauge the performance of the model
* Synthetic Minority Oversampling Techinque (SMOTE) - a technique to used when one data set dominates the other.
    We will be using a highly unballanced data set where the majority of the transactions are non-fraudelent,
    so SMOTE is used to balance it.

## Usage

- First, prepare the data for the model. In the command line type: 
```
    $> python prepData.py
```
- Second, train the model by running the line below in the command line. 
  You will get notified if the model file already exists. 
```
    $> python trainModel.py <num_of_epochs> <num_of_batches>
```
- Third, make the model predict fraudelent transactions by running from the command line:
```
    $> python detectCardFraud.py
```
## Team

| <a href="https://github.com/artsiom-sinitski" target="_blank">**Artsiom Sinitski**</a> |
| :---: |
| [![Artsiom Sinitski](https://github.com/artsiom-sinitski)](https://github.com/artsiom-sinitski)|
| <a href="https://github.com/artsiom-sinitski" target="_blank">`github.com/artsiom-sinitski`</a> |

## FAQ

## Support

Reach out to me at the following places:
- <a href="https://github.com/artsiom-sinitski" rel="noopener noreferrer" target="_blank">GitHub account</a>
- <a href="https://www.instagram.com/artsiom_sinitski/" rel="noopener noreferrer" target="_blank"> Instagram account</a>

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright ©2019 
