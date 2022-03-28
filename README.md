### Decision Tree Classifier From Scratch
<img src="https://64.media.tumblr.com/ca37102ccf7dd75783ab5fe393c77a84/tumblr_p2eonp6k261uuvpt3o1_500.gif" width="150"/>
I built a decision tree from scratch (python), using information gain as splitting criterion to classify poisonous and edible mushrooms. The information gain algorithm used is outlined <a href="https://www-users.cse.umn.edu/~kumar001/dmbook/ch3_classification.pdf">here.</a> This was my first introduction to using jupyter notebook, as well as data wrangling in python.

#### Data
The mushrooms dataset comes from the UCI Machine Learning Repository. Each instance has 22 attributes (all nominally valued), and is classfied as either poisonous (p) or edible (e). The data is split into a 1/3 test, 2/3 train split, where redundent variables and instances with missing values were removed from the data.

### Details on Implementation
#### Preparing/Cleaning the Data 
The data comes in two csv files, one with column names and the other with the data, each letter representing a particular class for its respective attribute. I first created the dataframe, then removed all useless data. This includes the _veil-type_ attribute, where all instances had the _partial veil type_, as well as all instances which had missing _stalk-root_ data. I then split up the data into its training and testing sets, and created a seperate numpy array version of these datasets for future use.

#### Functions for Training Procedure
There are a number of functions which aid in the main algorithm. Note, each column in the data represents a single attribute and its possible classes.
_check_mushroom_edibiity_ takes in an array of instances, and checks if all instances have the same classification of either poisonous or edible.
_get_possible_splits_ takes in an array of instances, and returns a list of lists of all possible classes for each attribute in the array.
_calculate_entropy_ takes in an array of instances, and calculates the entropy of the data on a given attribute.
_find_best_split_ takes in an array of instances and a list of possible splits (from the _get_possible_splits_ function), and finds the attribute in the array which gives the highest information gain, or the lowest entropy.
_classify_ takes in an array of instances, and tells us whether majority of the instances are poisonous or edible.

#### Main Algorithm
The main algorithm recursively iterates through the data, splitting on a single attribute until a maximum tree depth is achieved, or the data is perfectly classified. It stores the information in a dictionary, where the key-value pair is represented as the feature class it is splitting on, and a list its children nodes.

#### Model Evaulation and Reflection
After training and tuning the model, I tested it on the training set and found that the model had an extremely high accuracy. I also noticed that with a maximum tree depth of 2, there would be around 5/1862 misclassified instances (accuracy = 99.73%), while it would perfectly classify all data with a maximum tree depth of 3 (accuracy = 100%). This suggests the model is heavily overfitting the data and so I cannot generalise using this model.
If I were to make improvements to my model, I would use a 10 fold cross validation to train/test my model, as the small amount of data easily leads to overfitting. 
