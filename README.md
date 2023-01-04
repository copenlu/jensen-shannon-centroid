# Jensen-Shannon Centroid
A lightweight python package to calculate the Jensen-Shannon Centroid of a set of categorical distributions. The Jensen-Shannon Centroid is the minimizer Q of the equation:

$$\mathcal{L}(Q) = \sum_{m} \text{JS}(Q\\|p_{m})$$

where JS is the Jensen-Shannon divergence. We follow the ConCave–Convex procedure (CCCP) from Nielsen, 2020 to find Q.
   
> Nielsen, Frank. 2020. "On a Generalization of the Jensen–Shannon Divergence and the Jensen–Shannon Centroid" Entropy 22, no. 2: 221. https://doi.org/10.3390/e22020221

## Installation
Install directly using pip:

```
$ pip install jensen-shannon-centroid
```

## Dependencies
numpy

## Usage
There is one main method, `jensen_shannon_centroid.calculate_jsc`. This method takes in a list of size $M\times N\times K$ where:
- $K$ is the number of classes in each distribution
- $N$ is a set of different distributions each having $K$ classes
- $M$ are the different views of each distribution for which the Jensen-Shannon centroid will be calculated

The method will return an array of size $N\times K$, which are the Jensen-Shannon centroids of the set of $M$ different views of each of the $N$ distributions.

Here is an example usage:
```python
from jensen_shannon_centroid import calculate_jsc

distributions = [
    [[0.1, 0.9],
     [0.2, 0.8]],
    
    [[0.15, 0.85],
     [0.5,  0.5 ]]
  ]
  
calculate_jsc(distributions)

>>>returns: array([[0.12391947, 0.87608053],
                   [0.34213098, 0.65786902]])
```
