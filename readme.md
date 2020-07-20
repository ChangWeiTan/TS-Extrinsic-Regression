# Time Series Regression
This repository contains the source code for time series regression. 
We aim to learn the relationship between a time series and a scalar value.  

## Data
The archive containing time series regression datasets can be found at http://timeseriesregression.org/.
We recommend you to read the [paper](https://arxiv.org/abs/2006.10996) for a detailed discussion of the datasets and their sources.

## Code
The code is mainly divided as follows:
* The [demo.py](demo.py) file contains demo code for a single experiment run.
* The [run_experiments](run_experiments.py) file contains code to run a set of experiments.
* The [models](models) folder contains the models used for regression. 
* The [utils](utils) folder contains helper functions for the program.
* After each run, the results will be saved to the [output](output) folder.

## Reference
If you use any part of this work, please cite:
@article{Tan2020Time,
  title={Time Series Regression},
  author={Tan, Chang Wei and Bergmeir, Christoph and Petitjean, Francois and Webb, Geoffrey I},
  journal={arXiv preprint arXiv:2006.12672},
  year={2020}
}

## Acknowledgement
We appreciate the data donation from all the donors.
We would also like to thank [Hassan Fawaz](https://github.com/hfawaz/dl-4-tsc) for providing the base code for Deep Learning models and
[Angus Dempster](https://github.com/angus924/rocket) for providing the code for Rocket.
