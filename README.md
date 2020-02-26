**Summary**: CPC encodes the input into a sequence of representations, then predicts future observations from past ones. 
* Using a neural network $f_{\theta}$, encode each patch $x_{i, j}$ (of an image) into a feature vector $z_{i, j} =
  f_{\theta} (x_{i, j})$ to obtain a grid of feature vectors $z$. Then, apply a masked convolution network $g_{\phi}$ to
each feature vector to obtain a a context vector  $c_{i, j} = g_{\phi}(z_{i, j})$. 
    * Intuitively, each context vector is created by combining information from <u>only</u> the feature vectors above
      it. This is done using the masked convolution.
    * We end up with two grids: one of feature vectors and one of context vectors. 

* Use these rows of context vectors to linearly predict the rows of feature vectors below. 
    * $\hat{z}_{i+k, j} = W_{k}c_{i, j}$ where $W_{k}$ is a matrix and $k > 0$. 


* The goal is to correctly recognize the target $z_{i + k, j}$ among a set of *negative* samples $z_l$, taken from other
  locations in the image as well as other images in the minibatch. The probability of the correct target is calculated
with a softmax, and fed into cross-entropy loss. 

  
  $$
  \begin{align*}
  L_{CPC} &= -\sum_{i, j, k} \log p(z_{i + k, j} | \hat{z}_{i+k, j}, {z_l})
  \newline
  &= -\sum_{i, j, k} \log \frac{\exp(\hat{z}_{i+k, j}^{T}z_{i+k, j})}{\exp(\hat{z}_{i+k, j}^{T}z_{i+k, j}) + \sum_{l}
\exp(\hat{z}_{i+k, j}^{T}z_{l})}
  \end{align*}
  $$
