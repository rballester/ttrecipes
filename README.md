# ttrecipes 

**A cookbook of algorithms and tools that use the tensor train format**

*TT recipes* are routines for tensor train analysis, optimization and visualization via careful manipulation of the TT cores. We make heavy use of many key possibilities offered by the TT model (many are provided by the great [ttpy toolbox](https://github.com/oseledets/ttpy)):

- Compressing full and sparse tensors
- Elementary operations: sums and products of tensors, stacking, reshaping, etc.
- Recompressing existing tensors (TT-round algorithm)
- Adaptive cross-approximation and element-wise functions
- Mask tensors and TTs that behave similarly to deterministic finite automata

## References

- I. Oseledets. *Tensor-train decomposition* (2011)
- I. Oseledets, E. Tyrtyshnikov. *TT-cross approximation for multidimensional arrays* (2010)
- A. Cichocki, N. Lee, I. Oseledets, A.-H. Phan, Q. Zhao, D. P. Mandic. *Tensor Networks for Dimensionality Reduction and Large-scale Optimization: Part 1 (Low-Rank Tensor Decompositions)* (2016)
- R. Ballester-Ripoll, E. G. Paredes, R. Pajarola. *Sobol Tensor Trains for Global Sensitivity Analysis* (2017)