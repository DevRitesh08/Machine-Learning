# **Machine learning** 
**"Field of study that gives computers the ability to learn without being explicitly programmed "** -- Arthur samuel
# ML Algorithms 
1. Supervised Learning
    - it has seen the most rapid advancements
    - Used in most real world applications .
2. Unsupervised Learning 
3. Recommender System 
4. Reinforcement Learning

# **Supervised Learning** 

- It refers to the algorithms that learn x to y (x -> y) or input to output (input -> output) .
- It involves training a model using labeled data, where each input comes with a corresponding correct output.
- The goal of supervised learning is for the model to learn a mapping from inputs to outputs, so that when it is given new, unseen inputs, it can accurately predict the corresponding output labels.
- A fundamental concept in supervised machine learning is learning a class from examples. This involves providing the model with examples where the correct label is known, such as learning to classify images of cats and dogs by being shown labeled examples of both. The model then learns the distinguishing features of each class and applies this knowledge to classify new images.
![img.png](img.png)
- Examples :-
![img_1.png](img_1.png)

## Types of Supervised Learning in Machine Learning
Now, Supervised learning can be applied to two main types of problems:

1. Classification: Where the output is a categorical variable , Small number of possible outputs (e.g.,Breast Cancer Detection , spam vs. non-spam emails, yes vs. no).
2. Regression: Where the output is a continuous variable (e.g., predicting house prices, stock prices).

- ![img_2.png](img_2.png)
- ![img_6.png](img_6.png)

# **Unsupervised Learning** 

- It involves training a data using Unlabeled data .
- unsupervised learning algorithms are tasked with finding patterns and relationships within the data without any prior knowledge of the data's meaning .
- ![img_3.png](img_3.png)
- ![img_5.png](img_5.png)

## Unsupervised Learning Algorithms
There are mainly 3 types of Algorithms which are used for Unsupervised dataset.

1. **Clustering :** Clustering in unsupervised machine learning is the process of grouping unlabeled data into clusters based on their similarities. The goal of clustering is to identify patterns and relationships in the data without any prior knowledge of the data's meaning.
2. **Association Rule Learning :** This technique is a rule-based ML technique that finds out some very useful relations between parameters of a large data set. This technique is basically used for market basket analysis that helps to better understand the relationship between different products
3. **Dimensionality Reduction :** Dimensionality reduction is the process of reducing the number of features in a dataset while preserving as much information as possible. This technique is useful for improving the performance of machine learning algorithms and for data visualization.

- example of Clustering algorithm 
- ![img_7.png](img_7.png)

# Regression Model
- example :
- ![img_8.png](img_8.png)
#### **Important Terminology**
1. **Training Set** : Data used to train the model .
2. **x** : is the standard notation to denote the input variable or an input feature .
3. **y** : is the standard notation to denote the output variable or the target variable .
4. **m** : refers to the total number of training examples .
5. **(x,y)** : is used to define a single training example .
6. ![img_9.png](img_9.png)  here i is not exponent , it just the index in a training set and just refers to row i in the data table (x -> y) . 
- ![img_10.png](img_10.png)

#### **Imp** :
- ![img_11.png](img_11.png)
- y-hat is the prediction for y .
### **LAB**
- python is also zero indexed .
- To access a value in a Numpy array, one indexes the array with the desired offset. For example the syntax to access location zero of x_train is x_train[0].
- ![img_12.png](img_12.png)

```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000's of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
```

- Numpy arrays have a .shape parameter. x_train.shape returns a python tuple with an entry for each dimension. x_train.shape[0] is the length of the array and number of examples as shown below.

```python
# m is the number of training examples
# method 1
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
# method 2 , can also use the Python len() function as shown below.
m = len(x_train)
print(f"Number of training examples is: {m}")
```

![img_13.png](img_13.png)
```python
i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
```
![img_14.png](img_14.png)

```python
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()
```
## **Cost Function (J)**
- In order to implement linear regression the first key step is to define the Cost Function .
- ![img_15.png](img_15.png)
- Parameters in ML are variables (also called coefficients or weights) that we adjust during training to improve the model; here, "w" controls the slope and "b" the intercept.
- ![img_18.png](img_18.png)
- ![img_19.png](img_19.png)
- ![img_21.png](img_21.png)
1. let b = 0 then f(x) = wx
   - ![img_22.png](img_22.png)
   - ![img_23.png](img_23.png)
   - ![img_24.png](img_24.png)
   - the brown line shows the cost function curve .
   - blue line in vertical direction in the graphs indicate the error .
   - ![img_25.png](img_25.png)
   - The goal of linear Regression is to find the parameter w or w and b that results in the smallest possible value for the cost function J .
2. f(x) = wx + b
   - ![img_26.png](img_26.png)
   - The cost function curve when we have two parameters w and b will be like this
   - ![img_27.png](img_27.png) 
   - ![img_28.png](img_28.png)
   - it shows that different values of w and b will have same "J" as long as they lie on the same contour (concentric circle or ovals) , with the lowest value of "J" at the innermost oval .
   - ![img_29.png](img_29.png)
   - **Visualizing Examples**
     - ![img_30.png](img_30.png)
     - so if the value of w and b are from are a far outer oval then it will have a larger error .
     - ![img_31.png](img_31.png)
     - ![img_32.png](img_32.png)
     - perfect fit for the model given below :
     - ![img_33.png](img_33.png)

### **Assignment**
- ![img_34.png](img_34.png)

- ![img_35.png](img_35.png)
- Here, cost is a measure how well our model is predicting the target price of the house. The term 'price' is used for housing data.

## **Gradient Decent** : (imp .)
- In linear regression , we are not gonna manually try to read a contour plot for the best values of "w" and "b" ,because it isn't a good procedure and won't work once we get to more complex ML Models .
- So we need an efficient algorithm that we can write in code for automatically finding the values of parameters (w and b) and can provide us the best fit line  that minimizes the cost function "J" .
- This algorithm is known as "Gradient Decent" .
- Gradient Decent is an algo. that we can use to try to minimize any function , not just a cost function for linear regression .
- it applies to more general functions including other cost functions that work with models that have more than two parameters . ex. a cost function J with parameters w1,w2,w3...wn,b .
- for some functions J may not be bow or bowl shape , it is possible for their to be more than one possible minimum . ex. given below for the cost fun. of neural network
  - ![img_36.png](img_36.png)
### **Implementation**
- w = w − α⋅∂w(w,b)/∂w
- b = b − α⋅∂b(w,b)/∂b
- here α : alpha is the learning rate controls how big of a step we take when updating the model's parameters , w and b .
- so we'll simultaneously update both w and b until we reach convergence (point at local minimum where the parameters w and b no longer change much with each additional step )
- ![img_37.png](img_37.png)
- ![img_38.png](img_38.png)

### **Learning Rate (α)**

- What if the α is too small or too big ?
  1. if α is too small (like 0.000001) then we end up taking very small baby step's that will surely decrease the cost of J but incredibly slowly .
  2. if α is too large then it may never reach the minimum or overshoot .
- ![img_40.png](img_40.png)
- What if our parameter w is already at the local minima then the slope of secant is zero --> means the derivative of cost function is zero so w remains unchanged .
  - ![img_41.png](img_41.png)
  - it also explains why gradient descent reaches the local minimum , even with a fixed learning rate because as we reach the local minimum gradient decent will automatically take smaller steps . that's because as we approach local minimum , the derivative gets automatically smaller (angle decreases) .
  - ![img_42.png](img_42.png)

- ![img_43.png](img_43.png)
**Derivation of derivatives**
- ![img_44.png](img_44.png)
- ![img_46.png](img_46.png)
- ![img_47.png](img_47.png)

### Gradient Descent (summary)

- **How Gradient Descent Works:**  
  Gradient descent is an optimization algorithm used to minimize the cost function by iteratively updating the parameters (like `w` and `b`) in the direction of the steepest descent, i.e., the negative gradient.

- **Local vs. Global Minimum:**  
  In general, gradient descent may converge to a **local minimum** rather than the **global minimum**, depending on the initial values of the parameters. A *global minimum* is the point where the cost function `J` reaches its lowest possible value over the entire parameter space.

- **Linear Regression with Squared Error Cost Function:**  
  When using linear regression with a **squared error cost function**, the cost function is **convex**—it has a **bowl-shaped** curve. This means it has **only one global minimum** and **no local minima**.

- **Convex Functions:**  
  A convex function has a single global minimum and no other local minima. For such functions, **gradient descent is guaranteed to converge to the global minimum**, regardless of where the parameters are initialized.

- ![img_48.png](img_48.png)

## Multiple Input Variables 

![img_49.png](img_49.png)
![img_50.png](img_50.png)
![img_52.png](img_52.png)
![img_53.png](img_53.png)

## Vectorization

- While implementing a learning algorithm we can use vectorization to both shorten our code , and it will also increase its efficiency during execution .
- Advantages :
  - Makes code shorter .
  - fast execution of code  : because in background the numpy dot function  is able to ise parallel hardware in our computer , it works for both a normal computer (normal CPU) or if we are using a GPU ,  that is often used to accelerate machine learning jobs. 
- The numpy dot product operation is the vectorized implementation of the dot product  operation between two vectors . it is especially used when n is large , this will run much faster than any other way .
- ![img_54.png](img_54.png)

### Implementaton of vectorization 

- ![img_55.png](img_55.png)
- ![img_56.png](img_56.png)
- ![img_57.png](img_57.png)