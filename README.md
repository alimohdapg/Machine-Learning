# Machine-Learning
1. Introduction
In machine learning, classification is a supervised
learning technique used to identify the category of new
observations based on training data.

[1] This report
examines a binary classification task with partially
complete data coming from photos. The aim is to come up
with a machine learning method for classifying the photos
according to whether or not they are 'memorable'.
2. Approach
The machine learning approach chosen to tackle the
task this report aims to solve is random forest
classification. Random forest builds multiple decision
trees and merges them to get more accurate and stable
predictions. It also offers the following advantages which
make it a fitting choice for solving this task:
• Diversity – Not all features are considered while
making each tree, resulting in diverse trees.
• Works well with high dimensional data – The feature
space is reduced as a result of each tree not needing
to consider all features.
• Stability – The result is based on voting/averaging.[2]
It also relies on the following assumptions:
• Dataset isn’t large – Random forest relies on building
deep tress, large datasets, therefore, become
computationally expensive.
• Dataset isn’t noisy – Noisy datasets can result in
overfitting especially when the target classes
overlap.
• No formal distributions – Since random forest is a
non-parametric model and can handle skewed and
multi-modal data.[3]
3. Methods
3.1. The Provided Data
The data given for each photo consists of 4608 features
of which 4096 were extracted from a deep Convolutional
Neural Network (CNN) while the remaining 512 are gist
features. There are two sets of training data, training set 1
which contains 600 samples with all feature data present,
and training set 2 which has some missing feature data as
indicated by the value NaN (not a number).

The two training sets are given together with the ground
truth class labels, 1 for memorable and 0 for not
memorable. The class labels are assigned based on the
decision of 3 people, the labels are therefore also supplied
with confidence scores, 1.0 when all 3 people agree, and
0.66 when only the majority (2/3) agree on a label.
A second set of samples, the testing set, is also given
but without the labels. The goal is to obtain class label
predictions for photos in this set.
3.2. Training & Testing the Classifier

To train and test the classifier the repeated stratified k-
fold algorithm was used, configured with a k value of 10

(10 folds). The stratified version of k-fold was used to
ensure that each fold contains approximately the same
number of samples from each target class. A repeated
version of stratified k-fold was used to repeat the
validation process 3 times with a different randomization
in each repetition.
3.3. Model Selection
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

Student Number: 230936

Figure 1: The number of mussing features per sample in training
set 2. It can be inferred that all samples miss 800-1000 features.

2022 Fundamentals of Machine Learning Assignment Report.

2

200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249

250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
272
273
274
275
276
277
278
279
280
281
282
283
284
285
286
287
288
289
290
291
292
293
294
295
296
297
298
299

Number of
Estimators

Maximum
Features Accuracy % Standard
Deviation %
10 ‘sqrt’ 70.549 2.329
‘log2’ 69.167 2.096
50 ‘sqrt’ 74.422 2.117
‘log2’ 74.324 2.055
100 ‘sqrt’ 74.745 2.078
‘log2’ 74.255 2.172
200 ‘sqrt’ 75.000 2.187
‘log2’ 74.725 2.334

This shows that the best way to calculate the maximum
features is using ‘sqrt’ as that consistently gives better
results. Increasing the number of features also consistently
increases the accuracy. The values 300, 400, 500, 1000,
and 1500 were therefore also tried, this time only using
‘sqrt’ for the maximum features. The following were the
results:
Number of
Estimators Accuracy % Standard
Deviation %
300 75.088 2.282
400 75.137 2.078
500 75.304 2.062
1000 75.294 2.164
1500 75.206 2.094

As choosing 500 for the number of estimators
maximized accuracy while also minimizing the standard
deviation, it was chosen as the optimal value used to tune
the model.
3.4. Rescaling the Data
Two data scaling methods were tried. Normalization,
the process of scaling data to be in the range of 0 and 1, as
well as standardization, the process of scaling data so that
it has a mean value of 0 and a standard deviation of 1.
[4]
Combining standardization with PCA (principal
component analysis) to reduce the dimensions of the data
to 1 was also experimented with.
3.5. Feature Selection
Since two types of features were available (CNN and
gist features), the model was tested with both and each one
separately to see how they affect performance.

3.6. Imputing & Confidence Labels
To fill in the missing data, the algorithm k-nearest
neighbours was used. This algorithm predicts the missing
value by looking at the values of the neighbouring
columns (5 neighbours in this case) and subsequently
substitutes the missing value with the fittest value it
computed.
To evaluate the significance of confidence labels, the
effect discarding samples with a confidence label of 0.66
has on the model’s performance is also investigated.
4. Results & Discussion
4.1. Model Variations
Model
Variation Accuracy % Standard
Deviation %
All Features 75.245 2.225
CNN Features 74.333 2.063
Gist Features 68.206 2.762
Training Set 1
Only 77.278 4.821
Training Set 2
Only 73.929 1.700
Normalization 74.961 1.956
Standardization 75.255 2.082
PCA 59.559 2.241
Samples
w/Confidence
Label of 0.66
Discarded

78.420 3.961

The above table showcases the following:
• While CNN features are more important than gist
features, using both results in better accuracy.
• Using training set 1 alone results in a higher accuracy
but with a much higher standard deviation.
• Re-scaling the data (normalization and
standardization) doesn’t have a significant impact
on the model’s performance.
• Reducing the data’s dimensions to 1 using PCA
greatly reduces the model’s performance.
• Discarding samples with a confidence label of 0.66
results in a better-performing model.
As confidence-based sample discarding gave better
results, the classifier’s hyperparameters were retested with
discarding being a part of the model.

Table 1: The accuracy and standard deviation of the model’s
predictions using different values for the number of
estimators and maximum features.

Table 2: The accuracy and standard deviation of the model’s
predictions using different values for the number of estimators.

Table 3: The accuracy and standard deviation of the model’s
predictions using different model variations. Note: PCA also
included data standardization. Using all features and both
training sets was the default unless otherwise specified.

2022 Fundamentals of Machine Learning Assignment Report.

3

300
301
302
303
304
305
306
307
308
309
310
311
312
313
314
315
316
317
318
319
320
321
322
323
324
325
326
327
328
329
330
331
332
333
334
335
336
337
338
339
340
341
342
343
344
345
346
347
348
349

350
351
352
353
354
355
356
357
358
359
360
361
362
363
364
365
366
367
368
369
370
371
372
373
374
375
376
377
378
379
380
381
382
383
384
385
386
387
388
389
390
391
392
393
394
395
396
397
398
399

We can see that lowering the number of estimators
results in reduced accuracy and a higher standard
deviation. Higher numbers (200-1500) share
approximately the same performance, with the difference
between the highest and lowest accuracy being 0.594.
4.2. Final Model
The final variation of the model, used to obtain
predictions on the test dataset, used all features, 500
estimators, ‘sqrt’ for the maximum features, and discarded
samples with a confidence label of 0.66.
4.3. Future Improvements
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
References
[1] “Classification Algorithm in Machine Learning -
Javatpoint.” Www.javatpoint.com,

www.javatpoint.com/classification-algorithm-in-machine-
learning. Accessed 17 May 2022.

[2] E R, Sruthi. “Random Forest | Introduction to Random
Forest Algorithm.” Analytics Vidhya, 17 June 2021,

www.analyticsvidhya.com/blog/2021/06/understanding-
random-forest/. Accessed 17 May 2022.

[3] Richmond, Sarah. “Algorithms Exposed: Random Forest |

BCCVL.” BCCVL, 21 Mar. 2016, bccvl.org.au/algorithms-
exposed-random-forest/. Accessed 17 May 2022.

[4] Landup, David. “Feature Scaling Data with Scikit-Learn for
Machine Learning in Python.” Stack Abuse, 12 July 2021,

stackabuse.com/feature-scaling-data-with-scikit-learn-for-
machine-learning-in-python/. Accessed 18 May 2022.
