# ttrecipes 

*ttrecipes* is a Python library for working with, visualizing and understanding tensors (multiway arrays) compressed using the [tensor train format](http://epubs.siam.org/doi/abs/10.1137/090752286). We make heavy use of many key possibilities offered by the TT model (many are provided by the great [ttpy toolbox](https://github.com/oseledets/ttpy)):

- Compressing/decompressing full and sparse tensors ([```compression.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/core/compression.py), [```sparse.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/core/sparse.py))
- Operations on tensors: stacking, transposing, computing moments, etc. ([```util.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/core/util.py), [```analysis.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/core/analysis.py))
- Recompressing existing tensors (TT-round algorithm) ([```compression.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/core/compression.py))
- [Cross-approximation](http://www.mat.uniroma2.it/~tvmsscho/papers/Tyrtyshnikov5.pdf) wrapper for building tensors and surrogate modeling ([```cross.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/core/cross.py))
- Completion and regression ([```completion.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/core/completion.py))
- Sampling schemes for parameter spaces ([```sampling.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/core/sampling.py))
- Variance-based sensitivity analysis: Sobol indices, Shapley values, effective dimensions, etc. ([```sensitivity_indices.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/core/sensitivity_indices.py), [```sensitivity_analysis.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/sensitivity_analysis.py)) [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rballester/ttrecipes.git/master?filepath=examples%2Fsensitivity_analysis%2FSensitivity%20Analysis%20Examples.ipynb)
- Mask tensors and TTs that behave like deterministic finite automata ([```masks.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/core/masks.py), [```sets.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/core/sets.py))
- Visualization of TT tensors ([```mpl.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/mpl.py), [```tikz.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/tikz.py))
- A library of analytical models from physics, engineering, and computational science ([```models.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/models.py))

## Examples

Example of surrogate modeling interactive navigation (a [gradient boosting regressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) trained on the [*UCI Airfoil Self-Noise Data Set*](https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise), converted to the TT format via cross-approximation):

[<img src="https://github.com/rballester/ttrecipes/blob/master/images/airfoil_self_noise.png" width="768" title="Airfoil self-noise">](https://github.com/rballester/ttrecipes/raw/master/images/airfoil_self_noise.png)

[Sobol-based sensitivity analysis](http://onlinelibrary.wiley.com/book/10.1002/9780470725184) of a [10-dimensional fire-spread model](http://users.iems.northwestern.edu/~staum/ShapleyEffects.pdf) (~10 seconds were needed to compute these and more higher-order indices):

[<img src="https://github.com/rballester/ttrecipes/raw/master/images/fire_spread_sensitivity.png" width="768" title="Fire-spread sensitivity">](https://github.com/rballester/ttrecipes/raw/master/images/fire_spread_sensitivity.png)

See the [```examples/```](https://github.com/rballester/ttrecipes/tree/master/examples) folder for some sample scripts. Check out [this Jupyter Notebook](https://github.com/rballester/ttrecipes/blob/master/examples/sensitivity_analysis/Sensitivity%20Analysis%20Examples.ipynb) for the examples used in our paper [Tensor Approximation of Advanced Metrics for Sensitivity Analysis](https://arxiv.org/abs/1712.01633). You can also run and interact with the notebook online using Binder [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rballester/ttrecipes.git/master?filepath=examples%2Fsensitivity_analysis%2FSensitivity%20Analysis%20Examples.ipynb)


## Installation

_ttrecipes_ depends on [ttpy](https://github.com/oseledets/ttpy) and other common scientific computation packages (SciPy, NumPy, etc). The provided _environment.yml_, _requirements.txt_ and _ttpy\_repo.txt_ files collect these dependencies for an easy installation using __conda__ or __pip__ respectively. As usual, it is recommended to create a new Python environment for the project.

### Basic dependencies
If you use __conda__, you can create an environment (called _ttrecipes_ by default) with the required dependencies already installed with the following command:

    conda env create --file environment.yml

Or, if you prefer, you can also use __pip__ to install the required packages:

    pip install -r requirements.txt

Do not forget to activate the newly created environment with _'conda activate'_ or _'source activate'_.


### ttpy
__ttpy__ needs to be explicitly installed _after_ the basic dependencies. To install a version compatible with __ttrecipes__, run the following _pip_ command:

    pip install -r ttpy_repo.txt
    
**Important:** due to a bug in some releases of the _gfortran_ compiler in versions 5.0-8.0, the installation of __ttpy__ may fail ([issue #60](https://github.com/oseledets/ttpy/issues/60)). If this is your case, try to use a different _gfortran_ version (e.g. gfortran-4.8) or a different compiler (e.g. ifort).

### ttrecipes
Once the dependencies are ready, install __ttrecipes__ by cloning this repository and running _'pip install'_ with the current version in editable mode:

    git clone https://github.com/rballester/ttrecipes.git
    cd ttrecipes
    pip install -e .

## Project Structure

- There is a [```core```](https://github.com/rballester/ttrecipes/tree/master/ttrecipes/core) folder containing all lower-level utilities to work with TTs. They are all imported with a *horizontal structure*:

```
import ttrecipes as tr
tr.core.anyfunction()
```

- Higher-level functions are grouped as modules that have to be imported explicitly. Currently, there are:
    - [```mpl.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/mpl.py): TT visualization using *matplotlib*
    - [```tikz.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/tikz.py): TT visualization using *TikZ*
    - [```models.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/models.py): analytical functions for surrogate modeling, sensitivity analysis, etc.
    - [```sensitivity_analysis.py```](https://github.com/rballester/ttrecipes/blob/master/ttrecipes/sensitivity_analysis.py): high-level querying of Sobol indices, displaying and tabulating Sobol and other sensitivity metrics, etc.

For instance, use the following to visualize a TT tensor ([tt.vector object from ttpy](https://github.com/oseledets/ttpy/blob/develop/tt/core/vector.py)):

```
import ttrecipes.mpl
tr.mpl.navigation(t)
```

## Acknowledgment

This work was partially supported by the [UZH Forschungskredit "Candoc"](http://www.researchers.uzh.ch/en/funding/phd/fkcandoc.html), grant number FK-16-012.

## References

- I. Oseledets. [*Tensor-train decomposition* (2011)](http://epubs.siam.org/doi/abs/10.1137/090752286)
- I. Oseledets, E. Tyrtyshnikov. [*TT-cross approximation for multidimensional arrays* (2010)](http://www.mat.uniroma2.it/~tvmsscho/papers/Tyrtyshnikov5.pdf)
- A. Cichocki, N. Lee, I. Oseledets, A.-H. Phan, Q. Zhao, D. P. Mandic. [*Tensor Networks for Dimensionality Reduction and Large-scale Optimization: Part 1 (Low-Rank Tensor Decompositions)* (2016)](https://arxiv.org/abs/1609.00893)
- A. Cichocki, A.-H. Phan, Q. Zhao, N. Lee, I. V. Oseledets, M. Sugiyama, D. P. Mandic. [*Tensor Networks for Dimensionality Reduction and Large-Scale Optimizations. Part 2 Applications and Future Perspectives* (2017)](https://arxiv.org/abs/1708.09165)
- R. Ballester-Ripoll, E. G. Paredes, R. Pajarola. [*A Surrogate Visualization Model Using the Tensor Train Format* (2016)](https://dl.acm.org/citation.cfm?id=3002167)
- R. Ballester-Ripoll, E. G. Paredes, R. Pajarola. [*Sobol Tensor Trains for Global Sensitivity Analysis* (2017)](https://arxiv.org/abs/1712.00233)
- R. Ballester-Ripoll, E. G. Paredes, R. Pajarola. [*Tensor Approximation of Advanced Metrics for Sensitivity Analysis* (2017)](http://arxiv.org/abs/1712.01633)
- G. Rabusseau. [*A Tensor Perspective on Weighted Automata, Low-Rank Regression and Algebraic Mixtures* (2016)](http://pageperso.lif.univ-mrs.fr/~guillaume.rabusseau/files/phd_rabusseau_final.pdf)
