# Time Series Regression
This repository contains the source code for ``Time Series Regression``. 
We aim to learn the relationship between a time series and a scalar value. 
We note the similarity with a parallel field in the Statistics community known as 
``Scalar-on-Function-Regression (SoFR)`` and is working on implementing those methods.  

## Data
The archive containing 19 time series regression datasets can be found at [Monash UEA UCR Time Series Regression Archive](http://timeseriesregression.org/).
We recommend you to read the [paper](https://arxiv.org/abs/2006.10996) for a detailed discussion of the datasets and their sources.
For demo, please use the data in the data folder provided in this repository

## Code
The code is mainly divided as follows:
* The [demo.py](demo.py) file contains demo code for a single experiment run.
* The [run_experiments](run_experiments.py) file contains code to run a set of experiments.
* The [models](models) folder contains the models used for regression. 
* The [utils](utils) folder contains helper functions for the program.
* After each run, the results will be saved to the [output](output) folder.

### Dependencies
All python packages needed are listed in [requirements.txt](requirements.txt) file
and can be installed simply using the pip command. 

Some of the main packages are: 
* [keras](https://keras.io/)
* [matplotlib](https://matplotlib.org/)
* [numba](http://numba.pydata.org/)
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [scipy](https://www.scipy.org/)
* [tqdm](https://tqdm.github.io/)
* [xgboost](https://xgboost.readthedocs.io/en/latest/)

### Results
These are the results on the 19 Time series regression datasets from [Monash UEA UCR Time Series Regression Archive](http://timeseriesregression.org/).
The initial benchmark results in the [paper](https://arxiv.org/abs/2006.10996) showed that a simple linear model such as Rocket
performs best for the time series regression task. 
The full results can be obtained [here](http://timeseriesregression.org/data/ts_regression.xlsx).

![image](http://timeseriesregression.org/figures/ts_regression_cd.png)

## Reference
If you use any part of this work, please cite:
```
@article{Tan2020Time,
  title={Time Series Regression},
  author={Tan, Chang Wei and Bergmeir, Christoph and Petitjean, Francois and Webb, Geoffrey I},
  journal={arXiv preprint arXiv:2006.12672},
  year={2020}
}
```

## Acknowledgement
We appreciate the data donation from all the donors.
We would also like to thank [Hassan Fawaz](https://github.com/hfawaz/dl-4-tsc) for providing the base code for Deep Learning models and
[Angus Dempster](https://github.com/angus924/rocket) for providing the code for Rocket.
