# Hyperparameter_Tuning

Classify Raisins with Hyperparameter Tuning!
Use the techniques you have learned in this unit to classify different types of raisins.


# Grid Search
1.
We’ll be using the raisin data set from the UCI Machine Learning Repository. The task is to classify images of raisins as either Kecimen raisins or Besni raisins. The data set has seven numerical predictor variables.

We’ll use a support vector machine classifier to perform the classification. Create an SVC() model called svm.



2.
Scikit-learn’s SVC() allows you to choose several different support vector machine kernels by specifying the kernel parameter. We can also set the inverse of regularization strength by specifying the parameter C.

We’ll use the grid search algorithm to try out different values for kernel and C. Create a dictionary called parameters that we can use with GridSearchCV(). It should list 'linear', 'rbf', and 'sigmoid' as possible values for kernel. It should also list 1, 10, and 100 as possible values for C.


3.
Create a GridSearchCV() model called grid that will tune svm’s hyperparameters.


4.
Fit grid to X_train and y_train.

5.
Use the .best_estimator_ attribute to see what hyperparameters grid chose. Print the result.



6.
Copy and paste the following code to produce a summary table of grid search results.

df = pd.concat([pd.DataFrame(grid.cv_results_['params']), pd.DataFrame(grid.cv_results_['mean_test_score'], columns=['Score'])], axis=1)
 
cv_table = df.pivot(index='kernel', columns='C')
 
print(cv_table)

7.
Use GridSearchCV()’s .score() method to see how the model performs on the testing data.


# Bayesian Optimization
8.
Before continuing with the next tasks, you may want to comment out the lines of code that use grid. That way you won’t have to wait for it to be trained every time you run your code.

Let’s use Bayesian optimization to tune kernel and C. You’ll need to create a dictionary to specify prior distributions for each hyperparameter. Copy and paste the following code.

search_spaces = {'kernel': Categorical(['linear', 'rbf', 'sigmoid']), 'C': Real(1, 100, prior='uniform')}
9.
Create a BayesSearchCV model called bayes that will tune svm’s hyperparameters. Set n_iter to 10.

10.
Fit bayes to X_train and y_train.

11.
Use the .best_estimator_ attribute to see what hyperparameters bayes chose. Print the result. Is this different from what you found with grid search?


12.
Use BayesSearchCV()’s .score() method to see how the model performs on the testing data. How does this compare with the accuracy of the model obtained by grid search?


# TPOT

13.
Again, you may want to comment out the code from previous tasks to save computation time.

Create a TPOTClassifier model called tpot. Set generations to 2 and population_size to 20.

14.
Fit tpot to X_train and y_train.

15.
Use the .score() method to see how the TPOTClassifier model performs on the testing data. How does this compare with the accuracy of the models obtained by grid search and Bayesian Optimization?


Hint
Use something similar to what you used for corresponding tasks with GridSearchCV and BayesSearchCV.

16.
Use the .export() method to create a separate file called 'tpot_pipeline.py' that contains the pipeline created by TPOTClassifier.
