# Machine-Learning
## 1. Introduction
In machine learning, classification is a supervised
learning technique used to identify the category of new
observations based on training data.<sup>[1]</sup> This report
examines a binary classification task with partially
complete data coming from photos. The aim is to come up
with a machine learning method for classifying the photos
according to whether or not they are 'memorable'.
## 2. Approach
The machine learning approach chosen to tackle the
task this report aims to solve is random forest
classification. Random forest builds multiple decision
trees and merges them to get more accurate and stable
predictions. It also offers the following advantages which
make it a fitting choice for solving this task:
* Diversity – Not all features are considered while
making each tree, resulting in diverse trees.
* Works well with high dimensional data – The feature
space is reduced as a result of each tree not needing
to consider all features.
* Stability – The result is based on voting/averaging.<sup>[2]</sup>

It also relies on the following assumptions:
* Dataset isn’t large – Random forest relies on building
deep tress, large datasets, therefore, become
computationally expensive.
* Dataset isn’t noisy – Noisy datasets can result in
overfitting especially when the target classes
overlap.
* No formal distributions – Since random forest is a
non-parametric model and can handle skewed and
multi-modal data.<sup>[3]</sup>
## 3. Methods
### 3.1. The Provided Data
The data given for each photo consists of 4608 features
of which 4096 were extracted from a deep Convolutional
Neural Network (CNN) while the remaining 512 are gist
features. There are two sets of training data, training set 1
which contains 600 samples with all feature data present,
and training set 2 which has some missing feature data as
indicated by the value NaN (not a number).

<img width="500" alt="image" src="https://user-images.githubusercontent.com/84683922/184534745-8d978d1f-0d07-40af-8878-ed5679bd4834.png">

The two training sets are given together with the ground truth class labels, 1 for memorable and 0 for not
memorable. The class labels are assigned based on the
decision of 3 people, the labels are therefore also supplied
with confidence scores, 1.0 when all 3 people agree, and
0.66 when only the majority (2/3) agree on a label.

A second set of samples, the testing set, is also given
but without the labels. The goal is to obtain class label
predictions for photos in this set.
### 3.2. Training & Testing the Classifier

To train and test the classifier the repeated stratified k-
fold algorithm was used, configured with a k value of 10 (10 folds). The stratified version of k-fold was used to
ensure that each fold contains approximately the same
number of samples from each target class. A repeated
version of stratified k-fold was used to repeat the
validation process 3 times with a different randomization
in each repetition.
### 3.3. Model Selection
To initially select the best model, the random forest
classifier was tuned with different values for the number
of estimators and maximum features. The former is
concerned with the number of trees we want to build
before taking in the votes or averages for predictions,
while the latter is concerned with the number of features to
consider when looking for the best split.

For the maximum features two values were tried, ‘sqrt’
which takes the square root of the number of features, and
‘log2’ which takes log2(number of features). For the
number of estimators, the values 10, 50, 100, and 200
were tried. The following table shows the results this gave:

<img width="500" alt="image" src="https://user-images.githubusercontent.com/84683922/184534719-b7650e19-2944-4364-a9e0-bd0adcca9b67.png">

This shows that the best way to calculate the maximum
features is using ‘sqrt’ as that consistently gives better
results. Increasing the number of features also consistently
increases the accuracy. The values 300, 400, 500, 1000,
and 1500 were therefore also tried, this time only using
‘sqrt’ for the maximum features. The following were the
results:

<img width="500" alt="image" src="https://user-images.githubusercontent.com/84683922/184534795-1b0edf63-51da-4e9e-b2c1-dd5b21003628.png">

As choosing 500 for the number of estimators
maximized accuracy while also minimizing the standard
deviation, it was chosen as the optimal value used to tune
the model.
### 3.4. Rescaling the Data
Two data scaling methods were tried. Normalization,
the process of scaling data to be in the range of 0 and 1, as
well as standardization, the process of scaling data so that
it has a mean value of 0 and a standard deviation of 1.<sup>[4]</sup>
Combining standardization with PCA (principal
component analysis) to reduce the dimensions of the data
to 1 was also experimented with.
### 3.5. Feature Selection
Since two types of features were available (CNN and
gist features), the model was tested with both and each one
separately to see how they affect performance.

### 3.6. Imputing & Confidence Labels
To fill in the missing data, the algorithm k-nearest
neighbours was used. This algorithm predicts the missing
value by looking at the values of the neighbouring
columns (5 neighbours in this case) and subsequently
substitutes the missing value with the fittest value it
computed.

To evaluate the significance of confidence labels, the
effect discarding samples with a confidence label of 0.66
has on the model’s performance is also investigated.
## 4. Results & Discussion
### 4.1. Model Variations

<img width="500" alt="image" src="https://user-images.githubusercontent.com/84683922/184534878-62dbf8cf-0dba-4b8d-8b71-427d3d9ed76e.png">

The above table showcases the following:
* While CNN features are more important than gist
features, using both results in better accuracy.
* Using training set 1 alone results in a higher accuracy
but with a much higher standard deviation.
* Re-scaling the data (normalization and
standardization) doesn’t have a significant impact
on the model’s performance.
* Reducing the data’s dimensions to 1 using PCA
greatly reduces the model’s performance.
* Discarding samples with a confidence label of 0.66
results in a better-performing model.

As confidence-based sample discarding gave better
results, the classifier’s hyperparameters were retested with
discarding being a part of the model.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/84683922/184534942-4712824b-f26b-421c-8b1e-4de08d53a7ca.png">

We can see that lowering the number of estimators
results in reduced accuracy and a higher standard
deviation. Higher numbers (200-1500) share
approximately the same performance, with the difference
between the highest and lowest accuracy being 0.594.
### 4.2. Final Model
The final variation of the model, used to obtain
predictions on the test dataset, used all features, 500
estimators, ‘sqrt’ for the maximum features, and discarded
samples with a confidence label of 0.66.
### 4.3. Future Improvements
One of the tested model variations only used training set
1 and discarded samples with a confidence label of 0.66, it
achieved an accuracy of 89.681% but lacked a sufficiently
large sample size to be a reliable model (168 samples).
Therefore, having a larger training set that is not missing
any data and having more samples with a confidence label
of 1.0 would be greatly beneficial for improving the
model’s performance.

If time and computational cost weren’t issues, it would
have been beneficial to test out more combinations of
hyperparameters for both the random forest classifier and
the KNN imputer.
## References
[1] “Classification Algorithm in Machine Learning -
Javatpoint.” Www.javatpoint.com, www.javatpoint.com/classification-algorithm-in-machine-learning. Accessed 17 May 2022.

[2] E R, Sruthi. “Random Forest | Introduction to Random Forest Algorithm.” Analytics Vidhya, 17 June 2021, www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/. Accessed 17 May 2022.

[3] Richmond, Sarah. “Algorithms Exposed: Random Forest | BCCVL.” BCCVL, 21 Mar. 2016, bccvl.org.au/algorithms-exposed-random-forest/. Accessed 17 May 2022.

[4] Landup, David. “Feature Scaling Data with Scikit-Learn for Machine Learning in Python.” Stack Abuse, 12 July 2021, stackabuse.com/feature-scaling-data-with-scikit-learn-for-machine-learning-in-python/. Accessed 18 May 2022.
