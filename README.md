# Capstone_Starbucks
Udacity nano degree data science capstone 'Starbucks project'

**1. Installations**

Jupyter notebook

Libraries:

  Pandas
  
  Numpy
  
  Matplotlib
  
  Seaborn
  
  Sklearn
  
  Warnings
  
  Subprocess

**2. Project Motivation**

This project is part of the udacity Data Scientist Nanodegree training path. In this project we want to analyze three different datasets containing data from members of Starbucks. Our aim is to understand if offers can predict customer's behaviour. To do that, we will work with the provider datasets to discover features that help us understand the situation. Such as, demographic groups, differences between users, different offers and transactions. After that we will build a clasification model to understand how customers respond to the different offers. We will carry out an exploratory analysis of the datasets, cleaning of the data, create an ML model, train the model, evaluate and enhance the properties of the models and be able to predict the response of the users.

**3. Analysis**

We explored the three datasets to understand the information that we had, such as demographic and events information.
After this exploration we preprocessed and cleaned the data, to prepare it for apply a modeling strategy.
For the modeling phase we explored 6 different models: random forest, logistic regression, svm, naive bayes, decission trees and knn. After assessing all the models we decided to carry out further analysis with SVM and logistic regression. As this two models had an accuracy close to the 63%.
Then we tried to enhance the performance of the models with an hyperparameters refinement.After the refinement, the accuracy of the models didn't improve and we decided to continue exploring SVM.
We carried out a performance analysis of the SVM model with a confussion matrix, the ROC Curve and AUC index. The results showed a bias towards predicting non-completion. 

**4. Conclusions**

While the models showed some promise, the model had a slightly better-than-random chance of distinguishing between the two classes. The accuracy of around 63% suggested that the current approach is not yet reliable enough for business decision-making. To sumarize, further improvements and more exploration should be carried out to use this model to make predictions.

**5. File Descriptions**

Readme.md - Project description

Capstone_Starbucks_Post.md - Post with explanation of the challenge, procedure and conclusions.

portfolio.json - containing offer ids and metadata about each offer (duration, type, etc.)

profile.json—demographic data for each customer

transcript.json—records for transactions, offers received, offers viewed, and offers completed

Starbucks_Capstone_notebook.ipynb - Jupyter notebook with the code used for the analysis.

Starbucks_Capstone_notebook.html - Html file with the code used for the analysis to an easier reading.


**6. How to Interact with the project**

Upload all files and run them in a jupyter notebook console.
This project performs basic operations and transformations with dataframes. Such as, checking consistency, dropping nan values, transforming data types or creating new values based on existing data. It also uses matplot and seaborn to visualize the result of different operations. It also uses techniques from de Sklearn library.

**7. Licensing, Authors, Acknowledgements, etc.**

Datasets - Provided by Udacity thanks to Starbucks

Bibliography and consulted resources:

  -1. Scikit-learn Algorithms information. https://scikit-learn.org/stable/
  
  -2. Random forest classifier. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  
  -3. Logistic Refression. Details to implement and tume LR. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  https://www.youtube.com/watch?v=HYcXgN9HaTM
  
  -4. SVM. Model and implementation. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html 
  https://www.youtube.com/watch?v=_YPScrckx28
  https://www.youtube.com/watch?v=7sz4WpkUIIs
  
  -5. Naive Bayes. https://scikit-learn.org/stable/modules/naive_bayes.html
  
  -6. Decision trees. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
  
  -7. KNN. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
  
  -8. Cross validation, grid search. https://scikit-learn.org/stable/modules/cross_validation.html
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
  
  -9. ROC Curve. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
  
  -10. Auc. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
  https://www.youtube.com/watch?v=TEkvKx2tQHU
  https://www.youtube.com/watch?v=TmhzUdPpVPQ
  
  -11. Pandas.https://pandas.pydata.org/pandas-docs/stable/
  
  -12. Matplotlib.https://matplotlib.org/stable/contents.html
  
  -13. Seaborn.https://seaborn.pydata.org/
  
  -14. Others:
  
  Categorical conversion. https://www.youtube.com/watch?v=fyHaUMX9y0A
  Handling missing values. https://www.youtube.com/watch?v=uDr67HBIPz8
  Confussion matrix. https://www.youtube.com/watch?v=H2M3fT1njXQ

