
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

