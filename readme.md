# Time Series Extrinsic Regression
This repository contains the source code for ``Time Series Extrinsic Regression`` (TSER). 
We aim to learn the relationship between a time series and a scalar value. 
We note the similarity with a parallel field in the Statistics community known as 
``Scalar-on-Function-Regression (SoFR)`` and is working on implementing those methods.  

## Data
The archive containing 19 time series regression datasets can be found at [Monash UEA UCR Time Series Extrinsic Regression Archive](http://tseregression.org/).
We recommend you to read the [paper](https://arxiv.org/abs/2006.10996) for an overview of the datasets and their sources.

The `data` folder contains the actual feature definitions for each data set, as well as a sample data set that can be used for demo purposes.

## Models
The following models are implemented in this repository:
### Classical ML models 
1. Support Vector Regression (SVR) - A wrapper function for sklearn [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR) 
2. Random Forest Regressor (RF) - A wrapper function for sklearn [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor)
3. XGBoost (XGB) - A wrapper function for [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html) package
4. Linear Regression (LR) - A wrapper function for sklearn [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
5. Ridge Regression (Ridge) - A wrapper function for sklearn [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)

### Deep Learning for TSC 
1. Fully Convolutional Network ([FCN](https://github.com/hfawaz/dl-4-tsc))
2. Residual Network ([ResNet](https://github.com/hfawaz/dl-4-tsc))
3. Inception Time ([InceptionTime](https://github.com/hfawaz/InceptionTime))

### TSC models
1. Random Convolutional Kernels Transform ([Rocket](https://github.com/angus924/rocket))

## Features Transform
Some simple feature transformation have been implemented. 
1. Principal Component Analysis ([PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html))
2. Functional Principal Component Analysis ([FPCA](https://fda.readthedocs.io/en/latest/auto_examples/plot_fpca.html#sphx-glr-auto-examples-plot-fpca-py))
3. FPCA with BSpline Smoothing ([FPCA-BSpline](https://fda.readthedocs.io/en/latest/auto_examples/plot_fpca.html#sphx-glr-auto-examples-plot-fpca-py)) 

## Code
The code is mainly divided as follows:
* The [demo.py](demo.py) file contains demo code for a single experiment run.
```
Arguments:
-d --data_path      : path to dataset
-p --problem        : dataset name
-r --regressor      : name of the model
-t --transformer    : name of the transformer
-i --iteration      : iteration number
-n --normalisation  : normalisation (none, standard, minmax)
```
* The [run_experiments](run_experiments.py) file contains code to run a set of experiments.
* The [models](models) folder contains the models used for regression. 
* The [utils](utils) folder contains helper functions for the program.
* After each run, the results will be saved to the [output](output) folder.

## Dependencies
All python packages needed are listed in [requirements.txt](requirements.txt) file
and can be installed simply using the pip command. 

Some of the main packages are: 
* [keras](https://keras.io/)
* [matplotlib](https://matplotlib.org/)
* [numba](http://numba.pydata.org/)
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [scikit-fda](https://fda.readthedocs.io/)
* [sklearn](https://scikit-learn.org/stable/)
* [scipy](https://www.scipy.org/)
* [tqdm](https://tqdm.github.io/)
* [xgboost](https://xgboost.readthedocs.io/en/latest/)

## Results
These are the results on the 19 Time series regression datasets from [Monash UEA UCR Time Series Regression Archive](http://tseregression.org/).
The initial benchmark results in the [paper](https://arxiv.org/abs/2006.10996) showed that a simple linear model such as Rocket
performs best for the time series regression task. 
The full results can be obtained [here](http://tseregression.org/data/ts_regression.xlsx).

![image](http://tseregression.org/figures/ts_regression_cd.png)

## Reference
If you use any part of this work, please cite:
```
@article{
  Tan2020TSER,
  title={Time Series Extrinsic Regression}, 
  author={Tan, Chang Wei and Bergmeir, Christoph and Petitjean, Francois and Webb, Geoffrey I},
  journal={Data Mining and Knowledge Discovery},
  pages={1--29},
  year={2021},
  publisher={Springer},
  doi={https://doi.org/10.1007/s10618-021-00745-9}
}
```

## Acknowledgement
We appreciate the data donation from all the donors.
We would also like to thank [Hassan Fawaz](https://github.com/hfawaz/dl-4-tsc) for providing the base code for Deep Learning models and
[Angus Dempster](https://github.com/angus924/rocket) for providing the code for Rocket.
