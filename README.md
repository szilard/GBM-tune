
## Tuning GBMs (hyperparameter tuning) and impact on out-of-sample predictions

The goal of this repo is to study the impact of having one dataset/sample/"the dataset"  
when training and tuning machine learning models in practice (or in competitions) 
on the prediction accuracy on new data  
(that usually comes from a slightly different distribution due to non-stationarity).

To keep things simple we focus on binary classification, use only one source dataset 
with mix of numeric and categorical features and no missing values, we don't perform feature engineering,
tune only GBMs with `lightgbm` and random hyperparameter search (might also ensemble the best models), and 
we use only AUC as a measure of accuracy.

From the source data we pick 1 year of data for training/tuning and the following 1 year for testing (hold-out).
We take samples of give sizes (e.g. 10K, 100K, 1M records) from these. 

We choose a grid of hyperparameter values and take 100 random combinations from the grid.
For each hyperparameter combination we repeat the following resampling procedure 20 times:
Split the training set 80-10-10 into data used for (1) training (2) validation for early stopping
and (3) evaluation for model selection. 
We train the GBM models with early stopping and record the AUCs on the last split of data (3). We record 
the average AUC and its standard deviation.
Finally, we compute the AUC on the testset for the ensemble of the 20 models obtained
with resampling (simple average of their predictions).

We study the test AUC of the top performing hyperparameter combinations (selected based only on 
the information from the resampling procedure without access to the test set). In fact, we resample
the test set itself as well, therefore we obtain averages and standard errors for the test AUC.


### Train set size 100K records 

The evaluation AUC of the 100 random hyperparameter trials vs their ranking
(errorbars based on train 80-10-10 resampling):



The test AUC vs evaluation ranking (errorbars based on testset resampling):




Test vs evaluation AUC (with errorbars based on train 80-10-10 and test resampling, respectively):




A good choice of hyperparameters seems to be:
```
num_leaves = 1000
learning_rate = 0.03
min_data_in_leaf = 5
feature_fraction = 0.8
bagging_fraction = 0.8
```

For this combination, early stopping happens at `~200` trees in `~10 sec` for each resample (on a server with 16 cores/8 real cores) 
leaqding to evaluation AUC `0.815` and test AUC `0.745` (training data is coming from one given year, while the test
data is coming from the next year, therefore the decrease in prediction accuracy).

The runtime and number of trees for the different hyperparameter combinations vary, but the total training time
for the 100 random hyperparameter trials with 20 train resamples each is `~6 hrs`, while adding prediction time we
arrive at `~8 hrs` total runtime (the experiment can be easily parallelized to multiple servers as the trials in the random
search are completely independent).




