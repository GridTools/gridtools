#### Regular Accessors

Regular accessors are proxy objects
used to access [Data Field](data fields)
with any dimension. The access is performed by
specifying increments (offsets) with respect to
the current position of the stencil (iteration point).

As we know from
the [Storage](storage) documentation,
a data field can be a contiguous array
with arbitrary dimension, a vector of multidimensional arrays,
 or a kind of _matrix_ of multidimensional
 arrays (possibly with empty elements).
In order to avoid ambiguities we called
[Space Dimension](space dimension) the dimension of each
multidimensional array (or _snapshot_),
and [Field Dimension](field dimensions) the dimensions identifying
the position of the snapshot inside the
vector (or matrix). Since the field dimensions are atmost two,
we will identify them as _snapshot dimension_
and _component dimension_, in line with their most intuitive use case,
i.e. a vector representation of a time discretization.

The question from the
user API point of view is: how do we specify to the
accessor which dimension we want to access, or wether it is
a space dimension or a field dimension?

##### Space Dimensions

Let's start with an arbitrary dimensional array
(a single _snapshot_). The API exposed is very intuitive.
Suppose you have as first argument of the functor a
5D snapshot and an input accessor with null extent called ```acc```:
```c++
using acc = accessor<0, enumtype::in, extent<>, 5>;
```
We can access the 2 extra dimensions by specifying all the offsets
```c++
acc(0,0,-1,2,2)
```
We can also assign a name to a dimension, and increment it
using the following syntax
```c++
dimension<3> k; dimension<4> c; dimension<5> t;
acc(k-1, c+2, t+2)
```
In the latter API the order of the arguments is irrelevant
```c++
dimension<3> k; dimension<4> c; dimension<5> t;
acc(k-1, t+2, c+2) == acc(c+2, k-1, t+2)
```
Note that the second notation may greatly improve the readibility of the
user functor body by exposing a matlab-like API, especially when high
dimensionality is used.

##### Field Dimensions

Specifying an offset for a field dimension works exactly as for the
space dimension. So there is no way to distinguish the two only based
on the user functor. Whether we are accessing a space dimension or a field
dimension will depend only on the storage type which will be bound to the
accessor, and not on the accessor itself.

##### Accessor Alias

An accessor alias is a regular accessor which has an offset set at compile-time.
For instance, say you have a vector field in $\mathbb R^3$ with components h, v, w.
This vector field is accessed via an accessor called ```vec```
```c++
using vec = accessor<0, enumtype::in, extent<>, 4>;
```
However you may want to be able to refer to the third element of the vector
with ```w``` sometimes
in some expressions. You can do this defining an alias to the third component
of the accessor:
```c++
using w = alias<vec, dimension<4> >::set<2>;
```
The line above sets at compile-time the fourth offset to the value 2, so that we have
```c++
w() == vec(0,0,0,2)
```
which may contribute to considerably lighten the notation in complicated expressions.

##### Expressions

Remember the Do method example provided in [Example]?

```c++
template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = eval(lap(1, 0, 0)) - eval(lap(0, 0, 0));
            if (eval(out()) * (eval(in(1, 0, 0)) - eval(in(0, 0, 0))) > 0) {
                eval(out()) = 0.;
            }
```

We can notice that the ```eval``` keyword is repeated several times, which is somehow
tedious, especially when the expression is complicated it becaomes quickly very hard to read.
It is possible thought to embed the expressions in a single eval, i.e.
```c++
using namespace expressions;
template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = eval(lap(1, 0, 0) - lap(0, 0, 0));
            if (eval(out() * (in(1, 0, 0) - in(0, 0, 0)) > 0) {
                eval(out()) = 0.;
            }
```
This is achieved by using the expressions namespace, in which the operations ```+```, ```-```,
```*```, ```/```, ```pow<2>``` are
overloaded, and generate an expression to be evaluated. An example of its
usage, demonstrating its effectiveness, can be found in the
[Shallow Water](shallow water) example.

It is possible also to instantiate a compile time expression to be lazily evaluated,
useful for instance if we want to evaluate it multiple times
```c++
using namespace expressions;
constexpr auto cond = out() * (in(1, 0, 0) - in(0, 0, 0);
template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = eval(lap(1, 0, 0) - lap(0, 0, 0));
            if (eval(cond) > 0) {
                eval(out()) = 0.;
            }
```

#### Vector Accessors

Vector accessors are used when dealing with
[Expandable Parameters](expandandable parameters),
which are sequences of storages on which we want to perform the same
operations. They implement a "single stencil multiple storages" pattern,
as if the same stencil was applied to all the elements of the vector concurrently.

This "loop" or "vector operation" is completely abstracted away from the API of the
user function. The user has to define a _vector\_accessor_ as if it was a regular
accessor, and the corresponding stencil will be executed multiple times, each time
considering a different element in the vector.

NOTE: if multiple vector accessors are used in the same stencil, the corresponding
expandable parameters storages must have the same length

NOTE: we can mix vector accessors with regular accessors. In that case the regular
accessor will be the same for all the stencil invocations, while the vector accessor
will iterate over its components.

NOTE: the vector accessors are implemented using storage lists

For an example of usage of the vector accessor see the [Advection Pdbott](advection pdbott example)

#### Global Accessors

Global accessors are accessing read-only data which is independent of the current iteration point.
For this reason [Intent](intents), [Extent](extents) and [Offset](offsets) do not make sense for a global accessor.
Here the term "global" means that the data is the same for the whole grid. An example can be
a constant scalar parameter that you want to pass to the functor, or a user defined struct containing
various configuration options.

The API allows the user to define an arbitrary object deriving from [Global Parameter](global parameter), and pass it
to the computation. The accessor associated with this global parameter must be a global accessor

```c++
    using global_accessor< 0 > global_boundary;
```
Calling ```eval``` on the global accessor returns the user defined data structure. Supposing that
this data structure contains a user function called ```ordinal``` returning an integer, we can write
in the do method
```c++
    auto ordinal_ = eval(global_boundary()).ordinal();
```
NOTE: all the member functions defined in the user-defined data structure must be labeled with
GT_FUNCTION, in order for them to be callable from the device.

There is a special case for which we have a dedicated API: i.e. when the user defined object
(the global parameter)
defines parenthesis operator ```operator()```, and we want to call that operator from the Do method.
In that case the accessor's parenthesis operator can be used and the arguments will be
automatically forwarded to the global parameter. A typical example is the case in which we want to pass
a storage as a global parameter:
```c++
    using global_accessor< 0 > global_storage_;
    auto elem = eval(global_storage_(1,2,3));
```
A useful example to understand this use case can be found in the [Extended4D](extended4D example).