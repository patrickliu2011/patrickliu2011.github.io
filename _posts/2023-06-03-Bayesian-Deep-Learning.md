---
layout: post
title: Bayesian Deep Learning
---

Over the past few days, I've been doing some reading about Bayesian Deep Learning and Bayesian Neural Networks, so I'll be going over what I've learned.

# Bayesian Deep Learning and Bayesian Neural Networks
First, **Bayesian Neural Networks (BNNs)** are neural networks that use Bayesian properties, typically by having probability distributions over weights or activations, rather than deterministic values. What makes a BNN "Bayesian" is that the weights or activations are probability distributions. This is in contrast with classical, "frequentist" neural networks, in which weights and activations are deterministic values. Classical NNs can be seen as a special case of BNNs, where the probability distributions are degenerate (i.e. they are Dirac delta functions).

Why do we care about BNNs? Deep neural networks have been highly successful recently, but they are effectively black-boxes. As a result, it is difficult to quantify the uncertainty in their predictions, which is especially problematic when combined with the tendency of NNs to be overconfident in their predictions. In safety-critical applications like medical diagnosis, where being correct on average is not enough, this makes NNs much less deployable. The idea is that because BNN are designed to predict a distribution of possible outputs, their predictions naturally include uncertainty. There is some pushback against the idea that BNN posteriors inherently encodes model uncertainty, as in Jacob Buckman's blog post [Bayesian Neural Networks Need Not Concentrate](https://jacobbuckman.com/2020-01-22-bayesian-neural-networks-need-not-concentrate/).

The term **Bayesian Deep Learning (BDL)** seems to be used interchangeably with BNNs in most of the literature that I read. However, I will be using BDL in the sense defined in [A Survey on Deep Bayesian Learning (Wang et al. 2020)](https://arxiv.org/abs/1604.01662), as a general term that describes the unification of probabilistic graphical models (PGMs) and DL to involve a broader class of methods outside of BNNs under the BDL umbrella. Broadly, BDL allows us to combine the strengths of PGMs (e.g., uncertainty quantification, causal inference) and DL (e.g., predictive power).

In this post, I will go over how BNNs work, their advantages, and key challenges. I will then briefly go over the broader topic of BDL and a couple application areas.

* TOC 
{:toc}

# Bayesian Neural Networks
## Structure
In a structural sense, BNNs generally look like classical NNs, with the same types of layers (e.g., convolutional, fully-connected, recurrent). The key question is how handle weights that are distributions instead of numbers -- unlike classical NNs, forward propagation isn't simple multiplication of activation and weight matrices.

The key difference is that the weights and activations are probability distributions, rather than deterministic values. This means that the output of a BNN is a distribution over possible outputs, rather than a single output. This is in contrast with classical NNs, which output a single value. 

## Prediction
Mathematically, the goal of a network (in a supervised setup, at least) is to predict the true posterior distribution of the output $y$, conditioned on the corresponding input $x$ and the training data $D$. In a Bayesian view, we treat parameters as a distribution, where a particular set of parameter values can be more or less likely based on the data. With this in mind, the prediction we want to make is
$p(y \mid x, D) = \int_\theta p(y \mid x, θ')
p(\theta' \mid D) d\theta'.$

With complex models, this is computationally intractable -- it's infeasible to actually calculate the integral over the space of all possible parameters. However, it points us at the two calculations to learn: $p(y \mid x, \theta)$ and $p(\theta \mid D)$. $p(y \mid x, \theta)$ is represented by the forward pass of a model to make a prediction given a particular set of parameter values. In a BNN, this is equivalent to the forward pass of a classical NN. $p(\theta \mid D)$ is the likelihood of that set of parameter values, i.e. the distribution of parameters that we aim to learn. In practice, given these two values, we can sample values of $\theta$ to estimate the integral above.

## Learning
Because BNNs are not deterministic, optimizing their weights is more challenging than with classical CNNs. Because we are learning a distribution of weights, we need to use methods that can handle distributions, rather than just point estimates -- this process of learning the distribution given a network structure is called Bayesian inference. There are two main approaches to learning BNNs: **Monte Carlo Markov Chain (MCMC)** and **Variational Inference (VI)**.

### Markov Chain Monte Carlo
Markov Chain Monte Carlo (MCMC) methods involve constructing a Markov chain over all possible assignments of weights (i.e., a state in the Markov chain is an assignment of weights), where the stationary distribution (think stable state) of the Markov Chain is the true posterior distribution of the weights. Then, after running the Markov chain for a certain number of "burn-in" steps, any further samples will adhere to the distribution. Specifically, this is usually done with the Metropolis-Hasting algorithm, which probabilistically proposes the next state and then accepts or rejects it based off of the likelihood of the transition and the state. However, there are computational issues with MCMC methods:
- The number of "burn-in" steps needed to reach the stationary distribution is often very large, especially for complex models.
- to represent the posterior, we need to draw and store a large number of samples of the weights, especially because consecutive samples tend to be autocorrelated. 
Thus, although MCMC methods are guaranteed to converge to the true posterior, they are computationally expensive and difficult to scale to large models.

There are variations of MCMC methods designed to be more computationally efficient, such as Hamiltonian Monte Carlo (HMC) and No-U-Turn Sampling (NUTS). These methods make more efficient proposals for the next state in the Markov chain, which reduces the number of samples needed to converge to the true posterior. However, they are still computationally expensive and difficult to scale to large models.

### Variational Inference
Another approach to Bayesian infererence is variational inference (VI). Unlike MCMC, VI is not guaranteed to converge to the true posterior, but it is much more computationally efficient. The idea is to approximate the true posterior with a simpler parameterized distribution. For example, if we approximate the true distribution as a Gaussian, the estimate of the posterior would be parameterized by the mean and variance. Then we would optimize the parameters of that distribution to minimize the difference between the true posterior (which may not be Gaussian) and the approximation.

The typical optimization objective is to minimze the Kullback-Leibler (KL) divergence between the true posterior and the approximation. However, it is intractable to calculate the KL divergence with respect to the true posterior $p(\theta \mid D)$. Luckily, it turns out that minimizing the KL divergence is equivalent to maximize the **evidence lower bound (ELBO)**, which is a tractable function. Moreover, the **reparameterization trick** allows us to use backpropagation and stochastic gradient descent to optimize the ELBO. 

Here's how the reparameterization trick works. If we assume that all weights are normally distributed, then we can parameterize each weight distribution as its mean and variance. Suppose we have an normally distributed weight $w \sim \mathcal{N}(\mu, \sigma).$ We can reparameterize this weight as $w = \mu + \sigma \epsilon,$ where $\epsilon \sim \mathcal{N}(0, 1).$ Then all we have to store two numbers, $\mu$ and $\sigma$, and when we need to sample values of $w$ all we have to do is sample $\epsilon$ from the standard normal. Importantly, the reparameterization allows us to use backprop, since we do not need to backprop through any stochastic variables. We can therefore optimize the mean and variance of each variable using SGD.

*Side note:* If you know about variational autoencoders (VAEs), the reparameterization trick may look familiar. In fact, VAEs are technically a special case of BNNs, where only the latent activation layer -- the bottleneck between the encoder and decoder, which often represents the mean and covariance of a Gaussian in the latent space -- represents a distribution.

Although VI is much more computationally efficient than MCMC, it is *not guaranteed to converge to the true posterior*, especially if the prior distribution choice is not a good fit. For example, Monte Carlo Dropout is roughly equivalent to VI with a Bernoulli prior, which is not expressive enough to capture many posteriors. In practice, VI also tends to result in *underestimation of distribution variance*, resulting in over-confident models. This is a major issue because one of the main reasons to use BNNs is for their uncertainty quantification capabilities. 

Recent works in addressing these two challenges include normalizing flows for the former, and noise contrastive priors and calibration datasets for the latter.

### Side: Connection to Ensembles
**Ensembling** is the idea of training many separate models on the same data and combining their predictions (e.g. using a voting mechanism) to come up with an overall prediction. Each member of an ensemble can be seen as a sample from the parameter distribution, making an ensemble roughly approximate a batch of samples from the posterior distribution of weights.

# Bayesian Deep Learning
As I previously mentioned, Wang et al. 2020 uses BDL as a general term to refer to Bayesian systems with two components: perception and task-specific. The perception component's role is to process and learn from data, while the task-specific component's role is to incorporate prior information.

## Perception component
The perception component learns the structure of data. For example, in a BDL system that ingests video from a camera and usese it to decide a robot's actions, the perception component could be a model that processes input images and converts them into embedding vectors. They do not strictly need to be BNNs, but ideally include some probabilistic/uncertainty component to be compatible with Bayesian methods. Here are some of the many possible forms:
- Stacked denoising autoencoders (SDAEs) take in corrupted input data, map them to a lower dimensional latent space using an encoder, and try to predict the clean data using a decoder. SDAEs can be generalized to a probabilistic version to make them more compatible with Bayesian methods.
- Variational autoencoders (VAEs) map each input datapoint to a latent distribution and then try to reproduce the input data from the distribution again. 
- Natural Parameter Networks (NPNs) take in distributions rather than values as inputs, with weights as distributions as well.

## Task-specific component
The task-specific component generally handles probabilistic prior information. In the robot example in the previous example, the control system that handles deciding robot actions would be the task-specific component. A natural choice is to represent this information as a probabilistic graphical network (PGM) like a Bayesian network, which is relatively easy to reason over. Bayesian networks can be extended by using BNNs to model more complex relationships between variables. For example, a BNN can be used to represent all incoming edges of a node (in a graph $v_1 \rightarrow v_2 \leftarrow v_3$, a BNN could be trained to take $v_1$ and $v_3$ to predict $v_2$).

Beyond explicit Bayesian networks, stochastic processes can also be used to handle prior information. For example, the Wiener process models Brownian motion and is used to model topic evolution, while the Poisson process lends itself to modeling phoneme boundaries in ASR systems. Because these processes are Markovian, they can be seen as a sort of implicit Bayesian network.

# Example Applications of BDL
I read two articles on applying BNNs to healthcare and physics, which I will go over in this section.

## Healthcare
[A Review on Bayesian Deep Learning in
Healthcare: Applications and Challenges (Abdullah et al. 2022)](https://ieeexplore.ieee.org/abstract/document/9745083) does a survey of BNN use across tasks in healthcare. Currently, it seems that the primary usage of BNNs in healthcare is for uncertainty quantification, especially on computer vision tasks like image classification and segmentation. As such, there seems to be opportunity to use BDL for interpretability and causality and other tasks outside of single-image CV. 

After reading the review, I have a few questions of my own:
- Since the surveyed works mostly use MC-Dropout and VI as approximation methods instead of exact methods like MCMC, how accurate have their uncertainty estimates been in practice?
- Are the technical methods for BDL developed enough to perform any meaningful degree of causal inference beyond what is already known in healthcare?

## Physics
[Bayesian uncertainty quantification for machine-learned models in physics (Gal et al. 2022)](https://www.nature.com/articles/s42254-022-00498-4) is a discussion about Bayesian methods applied in physics. Specifically, although neural networks have strong predictive power, physics has a much higher bar for accurate estimates of uncertainty than most machine learning literature. Furthermore, BNNs are sensitive to prior choice, but incorporating physical first-principles in machine learning priors is still a challenging task.

An interesting insight I got from the article is about anomalies. In general, setting priors in BNNs can help with generalizing over outlying, out of distribution data. However, in physics, anomalous datapoints can also be rare, unexpected, and scientifically interesting events. Existing deep techniques are not particularly well-equipped to dealing with these anomalies.




