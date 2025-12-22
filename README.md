# Machine Learning Project

The objective of this machine learning project is to develop a predictive model based on a realistic dataset. Model construction can be implemented using either Python (with libraries such as Scikit-learn) or R (with packages mlr3 or tidymodels).

## Data files

This project involves building a predictive model to estimate an undisclosed characteristic related to the socio-economic status of 100090 individuals in Metropolitan France. The target variable for this prediction is only available in the learning set.

### Data structure

The data is distributed across several CSV files, divided into learning and test sets (50046 and 50044 individuals, respectively).

### Data handling considerations

This comprehensive dataset has a relatively complex structure. Some of the subsets contain missing values; for instance, the number of working hours in a job can be unknown. In addition, some information may be entirely missing for a given person. For instance, we may not know the last job of a retired person. These patterns of missingness are completely different: while imputation techniques may work for filling in a single missing value in the description of a person, imputing a large number of variables may lead to incorrect models. For instance when a person is not retired the most likely reasons for them to be missing from the job dataset are that they are not employee. They could be unemployed, they could also be independent contractors, etc. Considering their job data as "missing" would be completely wrong in this situation. In addition, the datasets may exhibit minor inconsistencies, such as a job position being presented as non-permanent in the job dataset but as permanent in the type of job dataset. It is recommended to implement some minimal sanity checks to verify the consistency of the data. If inconsistency corrections are needed, the main dataset should be considered more reliable than the job type dataset, which is, in turn, more reliable than the full job description dataset. Extreme care must be exercises when loading the data. In particular, some software may consider the INSEE city code (insee_code) as an integer and drop the leading 0 of some codes (e.g. turn 01001 into 1001). This may lead to an incorrect model.

## Expected Results

The goal of the project is to build a predictive model for the target variable given the other variables. More precisely, we are expected to

build a predictive model using the learning data;

estimate the future performances of the model on new data;

provide the prediction of our model on the test set.
