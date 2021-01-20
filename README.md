# Portfolio-Optimization-With-ML
Unsupervised learning techniques for stock diversification 

## Objective
Identify the most independent stocks using unsupervised learning techniques in order to maximise portfolio diversification.

## Approaches
Modern portfolio theory states that diversification of stocks is the most effective way to attain high returns with low risk. Several unsupervised learning approaches can be used to implement this principle effectively and improve upon traditional methods of portfolio optimization.

### 1. Graphical Network
The graphical network approach presents a major advantage of interpretability. A graphical network is constructed such that:
- Each node represents a different stock 
- Each edge weight represents the relationship between the stocks in terms of conditional independence

Representing the conditional independence structure allows us to ascertain the relationship between stocks independent of the general movement of the market.

### 2. Principal Component Analysis (PCA)
PCA aims to project correlated stocks onto an independent set of dimensions in order to find the stocks that maximise the variance explained.
Following eigen decomposition, the most signficant components are selected according to eigen value.

### 3. Clustering
Clustering in this context aims to choose a representative sample from a known population of stocks. Clustering aims to minimize the sum of distances from all points in a cluster to a median point, thus effectively clustering the stocks.

## Results

[Picture of graph before and after penalty loss function]
[Table comparing results]


## References
The main concepts implemented in this repository are based on the principles taught in the [Python and Machine Learning for Asset Management](https://www.coursera.org/learn/python-machine-learning-for-investment-management) course available on Coursera. 
