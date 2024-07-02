# genepro

<figure>
<img src="juniper_art.png" alt="art of a juniper, 'ginepro' in Italian" width=300px/>
<figcaption>
<i>Art of a juniper, "ginepro" in Italian, made with the</i> <a href="https://github.com/anopara/genetic-drawing">genetic drawing repo</a> <i>by</i> <a href="https://github.com/anopara">@anopara</a>.
</figcaption>
</figure>


## In brief

`genepro` is a Python library providing a baseline implementation of genetic programming, an evolutionary algorithm specialized to evolve programs.
This library includes a classifier and regressor that are compatible with scitik-learn (see **examples of usage** below).

Evolving programs are represented as trees.
The leaf nodes (also called *terminals*) of such trees represent some form of input, e.g., a feature for classification or regression, or a type of environmental observation for reinforcement learning.
The internal nodes represent possible atomic instructions, e.g., summation, subtraction, multiplication, division, but also if-then-else or similar programming constructs.

Genetic programming operates on a population of trees, typically initialized at random. 
Every iteration (called *generation*), promising trees undergo random modifications (e.g., forms of *crossover*, *mutation*, and *tuning*) that result in a population of offspring trees.
This new population is then used for the next generation.

<figure>
<img src="srfit.gif" width=400px alt="animation of genepro finding a symbolic regression solution">
<figcaption>
<i>
Example of 1D symbolic regression (made with <a href="https://gist.github.com/marcovirgolin/a83bb6e8fd634f9017586ab0c1605147">this gist</a>)
</i>
</figcaption>
</figure>

## Installation
For classification or regression, `genepro` relies only on a few libraries (`numpy`, `joblib`, and `scikit-learn`).
However, additional libraries (e.g., `gym`) are required to run the reinforcement learning example.
Thus, you can choose to perform a minimal or full installation.

### Minimal installation
To perform a minimal installation, run:
```
pip install genepro
```

### Full installation 
For a full installation, clone this repo locally, and make use of the file [requirements.txt](requirements.txt), as follows:
```
git clone https://github.com/marcovirgolin/genepro
cd genepro
pip install -r requirements.txt .
```

### Wish to use conda?
A conda virtual enviroment can easily be set up with:
```
git clone https://github.com/marcovirgolin/genepro
cd genepro
conda env create
conda activate genepro
pip install .
```



## Examples of usage

### Classification and regression
The notebook [classification and regression.ipynb](<classification and regression.ipynb>) shows how to use `genepro` for classification and regression, via scikit-learn estimators.

These estimators are intended for data sets with a small number of (relevant) features, as the evolved program can be written as a compact (and potentially **interpretable**) symbolic expression.


```
...
gen: 39,	best of gen fitness: -2952.999,	best of gen size: 46
gen: 40,	best of gen fitness: -2950.453,	best of gen size: 44
The mean squared error on the test set is 2964.646 (respective R^2 score is 0.512)
Obtained by the (simplified) model: 146.527 + -5.797*(-x_2**2 - 4*x_2 - 3*x_3 + 2*x_4 - x_5 - x_6*(x_4 - x_5) + x_6 - 5*x_8)
```
*Example of output of a symbolic regression model discovered for the [Diabetes data set](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)*.


### Reinforcement learning
The notebook [gym.ipynb](gym.ipynb) shows how `genepro` can be used to evolve a controller for the *CartPole-v1* environment of the OpenAI [gym](https://github.com/openai/gym) library.

<figure>
<img src="rand_n_evolved_cartpole.gif" width=600px alt="animation displaying a random cart pole controller">
<figcaption>
<i>Left: random cart pole controller; Right: evolved symbolic cart pole controller:

(x2 + x3) * (x2*x3 + x3 + x4 + 1) * log(abs(x2))^2 * log(abs(x3))^2 < 0.5? 'left' else 'right' </i>
</figcaption>
</figure>


## Citation
If you use this software, please cite it with:
```
@software{Virgolin_genepro_2022,
  author = {Virgolin, Marco},
  month = {9},
  title = {{genepro}},
  url = {https://github.com/marcovirgolin/genepro},
  version = {0.1.1},
  year = {2024}
}
```
