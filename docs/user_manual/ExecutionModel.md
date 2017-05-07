# GridTools Execution Model

Stencil operations are executed in a three dimensional index
space. This means that the iteration space (see [Concepts](CONCEPTS))
is three dimensional. The first two dimensions of the iteration space,
usually referred to as `I` and `J` dimensions identify the `IJ`
plane. There is no prescription on how the stencil operators in
different points of the `IJ` plane will be executed. Stencil operators
in the third dimension of the iteration space, usually referred as `K`
or vertical dimension, can have prescribe order of executions. There
are three different ways of executing on the `K` dimension

- `forward`: Index `k` in the vertical dimension is executed after index `k-1`, `0` is the first
- `backward`: Index `k` in the vertical dimension is executed after index `k+1`. `0` is the last
- `parallel`: No order is specified and execution can happen concurrently
