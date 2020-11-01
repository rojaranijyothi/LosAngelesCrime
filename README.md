## Predicting the Category of Crime that occurred given by time and location

### 1. Background,Problem and context:

Los Angeles, California is one of the largest metropolitan areas in the country, a port city, and incredibly ethnically diverse. It is within close proximity to the Mexican drug trade, home to the entertainment industry, and has residents across the socio-economic spectrum. Due to these and, a combination of other factors, Los Angeles has a colorful history of crime,including organized criminal activity, gang wars, riots and more. It is also important to note that the city has experienced an overall decline in crimes committed in the last several decades, according to the 
State of California Department of Justice and the Office of the Attorney General. In 2015, it was revealed that the Los Angeles Police Department had been under-reporting crime for eight years, making the crime rate in the city appear much lower than it really is.

Until recently crime prevention was studied based on strict behavioral and social methods, but the recent developments in Data Analysis have allowed a more quantitative approach in the subject. We will explore a dataset of nearly 9 years of crime reports from across all of Los Angeles neighborhoods, and we will create a model that predicts the category of crime that occurred, given the time and location.


### 2. Target Clients:

There are two types of clients that would be interested in the findings and predictive model from this project.

1. LosAngeles Police Department (that they can assess the type of crime ahead that's going to be occur in particular time and location ,and can take necessary actions or prevent it totally) 
2. US online or Newspapers that cover crime news that are driven by crime issues and backed by data analytics,to create awareness in the public.


### 3. Data Wrangling:

[Data Wrangling Notebook](https://github.com/rojaranijyothi/LosAngelesCrime/blob/master/Data%20Wrangling.ipynb)

#### 3.1 Data Collection

The source of the data set is:
https://data.lacity.org/A-Safe-City/Crime-Data-from-2010-to-2019/63jg-8b9z

This dataset contains 2.11M rows and 28 columns in csv format.

The following pdf which  contains the description for MO Codes is converted to csv and is used in this project:
https://data.lacity.org/api/views/63jg-8b9z/files/3db69cd3-446c-4dcd-82eb-3436dc08d3be?download=true&filename=MO_CODES_Numerical_20180627.pdf

#### 3.2 Data Definition

Description of columns

Column Name |	Description	| Type
----------- | ----------- | -----
DR Number |	Division of Records Number: Official file number made up of a 2 digit year, area ID, and 5 digits |	Plain Text
Date Reported |	MM/DD/YYYY |	Date & Time
Date Occurred |	MM/DD/YYYY |	Date & Time
Time Occurred	| In 24 hour military time |	Plain Text
Area ID |	The LAPD has 21 Community Police Stations referred to as Geographic Areas within the department. These Geographic Areas are sequentially numbered from 1-21 |	Plain Text
Area Name |	The 21 Geographic Areas or Patrol Divisions are also given a name designation that references a landmark or the surrounding community that it is responsible for |	Plain Text
Reporting District |	A four-digit code that represents a sub-area within a Geographic Area. All crime records reference the "RD" that it occurred in for statistical comparisons |	Plain Text
Crime Code |	Indicates the crime committed (Same as Crime Code 1) |	Plain Text
Crime Code Description |	Defines the Crime Code provided |	Plain Text
MO Codes |	Modus Operandi: Activities associated with the suspect in commission of the crime |	Plain Text
Victim Age |	Two character numeric |	Plain Text
Victim Sex |	F - Female M - Male X - Unknown |	Plain Text
Victim Descent |	Descent Code: A - Other Asian B - Black C - Chinese D - Cambodian F - Filipino G - Guamanian H - Hispanic/Latin/Mexican I - American Indian/Alaskan Native J - Japanese K - Korean L - Laotian O - Other P - Pacific Islander S - Samoan U - Hawaiian V - Vietnamese W - White X - Unknown Z - Asian Indian |	Plain Text
Premise Code |	The type of structure, vehicle, or location where the crime took place |	Plain Text
Premise Description |	Defines the Premise Code provided |	Plain Text
Weapon Used Code |	The type of weapon used in the crime |	Plain Text
Weapon Used Description |	Defines the Weapon Used Code provided |	Plain Text
Status Code |	Status of the case (IC is the default) |	Plain Text
Status Description |	Defines the Status Code provided |	Plain Text
Crime Code 1 |	Indicates the crime committed. Crime Code 1 is the primary and most serious one. Crime Code 2, 3, and 4 are respectively less serious offenses. Lower crime class numbers are more serious |	Plain Text
Crime Code 2 |	May contain a code for an additional crime, less serious than Crime Code 1 |	Plain Text
Crime Code 3 |	May contain a code for an additional crime, less serious than Crime Code 1 |	Plain Text
Crime Code 4 |	May contain a code for an additional crime, less serious than Crime Code 1 |	Plain Text
Address |	Street address of crime incident rounded to the nearest hundred block to maintain anonymity |	Plain Text
Cross Street |	Cross Street of rounded Address |	Plain Text
Location |	The location where the crime incident occurred. Actual address is omitted for confidentiality. XY coordinates reflect the nearest 100 block |	Location

#### 3.3 Data Cleaning

- Since the data column titles were mixed of upper and lowercase letters with space in between them ,we changed the original column names into a more accessible format to make data operations manageable.
- Dropping rows that contained NaN
- Typecasting columns
- Removing the outliers
- Extracting additional information from Time and Date Occurred such as hour, minute, day of the week, date, month

### 4. Exploratory Data Analysis

[EDA Notebook](https://github.com/rojaranijyothi/LosAngelesCrime/blob/master/ExploratoryDataAnalysis(EDA).ipynb)

Visualized the crimes that were commited from 2010-2019 based on various parameters.And also visualized victims sex, age, descent, weapons used, primary crime location, how crime type varies depends on location and time.

![yearly crimes](https://github.com/rojaranijyothi/LosAngelesCrime/blob/master/Figures/sex_descent.png)
![area_year](https://github.com/rojaranijyothi/LosAngelesCrime/blob/master/Figures/area_year.png)
![age abvoe 70](https://github.com/rojaranijyothi/LosAngelesCrime/blob/master/Figures/age_70.png)

### 5. Data Preprocessing

[Preprocessing Notebook](https://github.com/rojaranijyothi/LosAngelesCrime/blob/master/Preprocessing.ipynb)

 - Feature extraction
 - Stratified Sampling
 - Handling Rare Categories
 - Create dummy features for categorical variables - Binary encoding
 - Stratified Train-Test Splits
 - Scale the data to prepare for model creation
 
 ### 6. Modeling
 
 [ML Notebook](https://github.com/rojaranijyothi/LosAngelesCrime/blob/master/Modeling.ipynb)
 
 In this part,we are trying to build two models
   -  1. Multiclass classification for all the crime types including low frequency crimes.
   -  2. Multiclass classification for high frequency(top 10) crimes.
 
Since target is a categorical variable with many classes and data is labelled data,it is a Supervised multi-class classification problem.The dataset is imbalanced because the classes in target have an unequal distribution.For modeling I choose to work with a machine learning library - scikit.learn.

Metrics: Choosing the right metrics is the key to assess the performance of a model. I choose to take “weighted” F1 score.For multi-class problems with imbalance data,we have to average the F1 scores for each class.The weighted F1 score averages the F1 score for each class by taking the class imbalances into account.In other words,the number of occurrences of each class does figure into calculation when using “weighted” score.

#### 1. Multiclass classification for all the crime types including low frequency crimes

1. Tried imbalanced classification techniques to spot check:
  - Cost-Sensitive Algorithms
  - Data Resampling Algorithms(SMOTETomek)
2. Tried another method with a combination of supervised and unsupervised learning - Classification with DBSCAN(Anomaly detection and Clustering).

Choosing the best model:
 
Model | Algorithm | f1_score(weighted) | Precision | Recall | Training | Testing
----- | --------- | ------------------ | --------- | ------ | -------- | -------
Without Cluster | Random Forest | 0.665 | 0.687 | 0.692 | 1.000 | 0.692
Without Cluster | CostSenstivity | 0.684 | 0.712 | 0.705 | 1.000 | 0.705
Without Cluster | SMOTETomek | 0.708 | 0.719 | 0.720 | 0.992 | 0.720
With Cluster | Random Forest | 0.662 | 0.6815 | 0.689 | 1.000 | 0.689
With Cluster | CostSenstivity | 0.678 | 0.705 | 0.701 | 1.000 | 0.701
With Cluster | SMOTETomek | 0.703 | 0.713 | 0.716 | 0.992 | 0.716

I choose CostSensitivity(Without Cluster) as final model eventhough SMOTETomek outperforms CostSensitivity,because SMOTETomek is computationally so expensive.
After tuning the parameters,achieved performance accuracy **F1-score(weighted) - 0.71**.

Because the target has 76 classes,it would not be possible to visualize the predictions through confusion matrix.But we can review the predictions through Reciever Operating Characteristic(ROC).The AUC-ROC curve is only for binary classification problems. But we can extend it to multiclass classification problems by using the One vs All technique.

![AUC-ROC](https://github.com/rojaranijyothi/LosAngelesCrime/blob/master/Figures/aucroc_model1.png)

#### 2. Multiclass classification for high frequency(top 10) crimes

Since we choose only the high frequency(top 10) crimes, all the classes in the response variable have merely an equal distribution.
I evaluated the following machine learning models :

  - Support Vector Machine (SVM)
  - k-Nearest Neighbors (KNN)
  - LogisticRegression (LogReg)
  - Random Forest (RF)
  - XGBoost(XGB)

The output results are:

Model | Accuracy(mean) | Accuracy(std)
----- | -------------- | -------------
SVM | 0.656 | 0.0041
KNN | 0.562 | 0.0072
RF | 0.692 | 0.0053
LogReg | 0.604 | 0.0084
XGB | 0.708 | 0.008 

Again here I am a little biased to RandomForest because of its fast runtime speed.So I choose RF as our model even though XGB outperforms RF.I built a model using RandomForest with StratifiedKFold(5 splits) and achieved a f1-score  as 0.831.After that I did hyperparameter tuning for best parameters to improve the performance of our model.With the best parameters that we got after tuning,we were able to increase the accuracy by 1% in other words achieved a pretty good accuracy **F1-score - 0.84**.

For this model we can we visualize the results through confusion matrix as well as with AUC-ROC.

**PREDICTIONS**

![AUC-ROC](https://github.com/rojaranijyothi/LosAngelesCrime/blob/master/Figures/aucroc_model2.png)

![Confusion matrix](https://github.com/rojaranijyothi/LosAngelesCrime/blob/master/Figures/confusion%20matrix.png)

In conclusion, our work led to interesting results, analysis and statistics, but also provided useful tools both for authorities and population, which allows a better understanding of crimes in LA.

This project gave me an opportunity to explore this freely available dataset using a proper data science pipeline of data wrangling, data analysis, data visualization, prediction, and data storytelling.

### 7. Future Improvements

 - In order to boost the classification accuracy, it is necessary to incorporate other information like Modus Operandi(MO) which we did not use in our model.Additionally, some events and the outcomes of the events may be associated with some crime types.Events information and weather information can also be incorporated.It will be interesting to see whether these features can help the classification.
 - We could use a Clustering algorithm, namely K-Modes Clustering, which is similar to K-Means, but uses modes instead of means, making itself usable for cluster computation on categorical variables. As a result of the clustering process, we could figure out some cluster centroids which can potentially be interesting for authorities, since they can indicate "standard frequent crimes", allowing authorities to concentrate their forces in order to contrast crimes indicated by  centroids and similar ones.
 - Due to memory constraints on Jupyter notebook, I had to train a sample of size 20000 from the original dataset. Without resource limitations, I would love to train on the full dataset. Preliminary tests showed that the bigger the training size, the higher the accuracy.





