## EGGROLL MNIST

If you just wanna run the experiments, run `./run_experiments.sh`.

There is two scripts here: mnist_backprop.py and mnist_eggroll.py.   

The first uses standard backpropagation to converge on the MNIST dataset. The latter uses backprop-free Evolution Strategies (ES) to converge on the MNIST dataset.   

Both scripts use the same network architecture (3-layer MLP with 256 hidden units), same activation function (GELU), same batch size (128), same number of epochs (10). The remaining hyperparameters of both were optimized in extensive grid searches.

### Results:
Both achieve around 97.5% accuracy on the test set. EGGROLL requires substantially more memory and is about 100x slower than backpropagation.  

This is only a proof of concept though and the implementation of EGGROLL is quite naive. I'm working on making it much more memory and compute efficient.   

The most interesting part then is the flexibility of ES / EGGROLL. E.g. it already uses the raw training accuracy as its signal (fitness score). It is also possible to use any other signal, or any components in the network, regardless of differentiability because ES does not require gradients.

### Background:
I implemented the main algorithm (EGGROLL) from this paper:
**"Evolution Strategies at the Hyperscale"**  
*Bidipta Sarkar, Mattie Fellows, Juan Agustin Duque, et al.* (2025)  
[https://arxiv.org/pdf/2511.16652](https://arxiv.org/pdf/2511.16652)

![EGGROLL Visualization](visualizations/vis1.png)  
![EGGROLL Pseudocode](visualizations/vis2.png)  

EGGROLL is a Evolution Strategies (ES) algorithm that achieves much higher efficiency than naive ES. 

ES is based on random perturbations applied to the weight matrices (the number of independent perturbations is given by the "population size"). These are then used in separate rollouts in the task at hand and evaluated with a fitness score. The weight update moves the weights in the direction of the "most successful" perturbations (this is a highly simplified explanation).

The main problem with naive ES is the memory and compute inefficiency which is O(N) where N = population size. 

The authors of the EGGROLL paper solved this problem to some extent by using low-rank perturbations that no longer involve materializing the full-rank matrix perturbations at any point in time. Read the paper or my code if you wanna see how exactly this is done and some neat tricks they used to make it work.

The authors of the paper used the above mentioned flexibility of ES to do INT8 native training. I also replicated that and my results were pretty bad. Obviously, the lower precision makes learning much harder. But also, the overhead of clipping and casting is quite high. I'm not too familiar with optimizing INT8 which might be part of the problem. But I'm more interested in other uses of the flexibility of ES and getting the low-rank technique super fast.  

Write me an e-mail if you wanna chat about this. I'm Johann, my e-mail is johann.zahlmann@gmail.com

