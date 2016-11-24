
Lazy Learning – Classification Using Nearest Neighbors
-------------------------------------------------------

- Computer vision applications, including optical character recognition and facial recognition in both still images and video 
- Predicting whether a person will enjoy a movie or music recommendation 
- Identifying patterns in genetic data, perhaps to use them in detecting specific proteins or diseases

Strengths
- Simple and effective
- Makes no assumptions about the underlying data distribution
- Fast training phase

Weaknesses
- Does not produce a model, limiting the ability to understand how the features are related to the class
- Requires selection of an appropriate k
- Slow classification phase
- Nominal features and missing data require additional processing

Nearest neighbor classifiers are well-suited for classification tasks, where relationships among the features and the target classes are numerous, complicated, or extremely difficult to understand, yet the items of similar class type tend to be fairly homogeneous. Another way of putting it would be to say that if a concept is difficult to define, but you know it when you see it, then nearest neighbors might be appropriate. On the other hand, if the data is noisy and thus no clear distinction exists among the groups, the nearest neighbor algorithms may struggle to identify the class boundaries.


Probabilistic Learning – Classification Using Naive Bayes
----------------------------------------------------------

- Text classification, such as junk e-mail (spam) filtering 
- Intrusion or anomaly detection in computer networks 
- Diagnosing medical conditions given a set of observed symptoms

Strengths
- Simple, fast, and very effective 
- Does well with noisy and missing data 
- Requires relatively few examples for training, but also works well with very large numbers of examples 
- Easy to obtain the estimated probability for a prediction

Weaknesses
- Relies on an often-faulty assumption of equally important and independent features 
- Not ideal for datasets with many numeric features 
- Estimated probabilities are less reliable than the predicted classes

Bayesian classifiers are best applied to problems in which the information from numerous attributes should be considered simultaneously in order to estimate the overall probability of an outcome. While many machine learning algorithms ignore features that have weak effects, Bayesian methods utilize all the available evidence to subtly change the predictions. If large number of features have relatively minor effects, taken together, their combined impact could be quite large.

Divide and Conquer – Classification Using Decision Trees
---------------------------------------------------------

- Credit scoring models in which the criteria that causes an applicant to be rejected need to be clearly documented and free from bias 
- Marketing studies of customer behavior such as satisfaction or churn, which will be shared with management or advertising agencies 
- Diagnosis of medical conditions based on laboratory measurements, symptoms, or the rate of disease progression

Strengths
- An all-purpose classifier that does well on most problems 
- Highly automatic learning process, which can handle numeric or nominal features, as well as missing data 
- Excludes unimportant features 
- Can be used on both small and large datasets 
- Results in a model that can be interpreted without a mathematical background (for relatively small trees) 
- More efficient than other complex models

Weaknesses
- Decision tree models are often biased toward splits on features having a large number of levels - It is easy to overfit or underfit the model 
- Can have trouble modeling some relationships due to reliance on axis-parallel splits 
- Small changes in the training data can result in large changes to decision logic 
- Large trees can be difficult to interpret and the decisions they make may seem counterintuitive

Decision trees are perhaps the single most widely used machine learning technique, and can be applied to model almost any type of data— often with excellent out-of-the-box applications. 
This said, in spite of their wide applicability, it is worth noting some scenarios where trees may not be an ideal fit. One such case might be a task where the data has a large number of nominal features with many levels or it has a large number of numeric features. These cases may result in a very large number of decisions and an overly complex tree. They may also contribute to the tendency of decision trees to overfit data. Even this weakness can be overcome by adjusting some simple parameters.

Divide and Conquer – Classification Using Decision Rules
---------------------------------------------------------

- Identifying conditions that lead to a hardware failure in mechanical devices
- Describing the key characteristics of groups of people for customer segmentation
- Finding conditions that precede large drops or increases in the prices of shares on the stock market

Strengths

Weaknesses

Rules can be generated using decision trees. Decision trees bring a particular set of biases to the task that a rule learner avoids by identifying the rules directly. 
Rule learners are generally applied to problems where the features are primarily or entirely nominal.
They do well at identifying rare events, even if the rare event occurs only for a very specific interaction among feature values.



