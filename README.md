# ttrecipes 

*TT recipes* are routines for [tensor train](http://epubs.siam.org/doi/abs/10.1137/090752286) analysis, optimization and visualization via careful manipulation of the TT cores. We make heavy use of many key possibilities offered by the TT model (many are provided by the great [ttpy toolbox](https://github.com/oseledets/ttpy)):

- Compressing full and sparse tensors
- Elementary operations: sums and products of tensors, stacking, reshaping, etc.
- Recompressing existing tensors (TT-round algorithm)
- [Cross-approximation](http://www.mat.uniroma2.it/~tvmsscho/papers/Tyrtyshnikov5.pdf) wrapper for surrogate modeling
- Completion and regression
- Global variance-based sensitivity analysis and Sobol indices
- Mask tensors and TTs that generalize deterministic finite automata

## Dependencies

- NumPy
- [ttpy](https://github.com/oseledets/ttpy)

## Installation

The current version of the _ttpy_ toolbox in PyPI has some incompatibilities with _TT recipes_. The provided _requirements.txt_ file installs a valid version from its GitHub [repository](https://github.com/oseledets/ttpy). The recommended way to install _TT recipes_ and its dependencies is:

    git clone https://github.com/rballester/ttrecipes.git
    cd ttrecipes
    pip install numpy cython
    pip install -r requirements.txt

## Examples

See the ```examples/``` folder for some sample scripts. Check out [this Jupyter Notebook](https://github.com/rballester/ttrecipes/blob/master/examples/sensitivity_analysis/Sensitivity%20Analysis%20Examples.ipynb) for the examples used in our paper *Tensor Approximation of Advanced Metrics for Sensitivity Analysis*.

You can also run it on Microsoft Azure Notebooks (a free account is needed): [![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/egparedes/libraries/ttrecipes/html/examples/sensitivity_analysis/Sensitivity%20Analysis%20Examples.ipynb)

## References

- I. Oseledets. [*Tensor-train decomposition* (2011)](http://epubs.siam.org/doi/abs/10.1137/090752286)
- I. Oseledets, E. Tyrtyshnikov. [*TT-cross approximation for multidimensional arrays* (2010)](http://www.mat.uniroma2.it/~tvmsscho/papers/Tyrtyshnikov5.pdf)
- A. Cichocki, N. Lee, I. Oseledets, A.-H. Phan, Q. Zhao, D. P. Mandic. [*Tensor Networks for Dimensionality Reduction and Large-scale Optimization: Part 1 (Low-Rank Tensor Decompositions)* (2016)](https://arxiv.org/abs/1609.00893)
- A. Cichocki, A.-H. Phan, Q. Zhao, N. Lee, I. V. Oseledets, M. Sugiyama, D. P. Mandic. [*Tensor Networks for Dimensionality Reduction and Large-Scale Optimizations. Part 2 Applications and Future Perspectives* (2017)](https://arxiv.org/abs/1708.09165)
- R. Ballester-Ripoll, E. G. Paredes, R. Pajarola. [*A Surrogate Visualization Model Using the Tensor Train Format* (2016)](https://dl.acm.org/citation.cfm?id=3002167)
- R. Ballester-Ripoll, E. G. Paredes, R. Pajarola. [*Sobol Tensor Trains for Global Sensitivity Analysis* (2017)](https://arxiv.org/abs/1712.00233)
- R. Ballester-Ripoll, E. G. Paredes, R. Pajarola. [*Tensor Approximation of Advanced Metrics for Sensitivity Analysis* (2017)](http://arxiv.org/abs/1712.01633)
- G. Rabusseau. [*A Tensor Perspective on Weighted Automata, Low-Rank Regression and Algebraic Mixtures* (2016)](http://pageperso.lif.univ-mrs.fr/~guillaume.rabusseau/files/phd_rabusseau_final.pdf)
