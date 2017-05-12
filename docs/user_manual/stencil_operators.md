# Stencil Operators

*Stencil operators* are the equivalent of _functors_ in regular C++
 code, or they may be considered to be the _GridTools functions_. They
 are assumed to have no side-effects and no status (this is why they are marked as `static`). As fuctions they have an
 _interface_ and an _implementation_. The interface informs both the caller,
 on the order and types of arguments that have to be passed to it, and
 the implementation, on the names and types of the symbols available
 to it.

The stencil operator specifies the computation to be performed in each
point of the [Iteration Space](iteration space) of the stencil
computation (see [stencil composition documentation]). In the
implementation, a point of the [Iteration Space](iteration space) at
which the stencil operator is called is referred to as *point of
evaluation*.

A stencil operator is a `class`, or a `struct`, with the following
public properties:

- A list of *accessors types* that are associated to the
  [Data Field](data field)s the stencil operator will access in its
  implementation.
- An `arg_list` listing all the accessors types defined above
- A set of *static template member functions* named `Do`, also
  referred to as _`Do` methods_. To run on GPUs the function should
  also be a [GT_FUNCTION](`GT_FUNCTION`).

See the [end of this Section for an example](#example).

## Stencil Operator Interface

### Accessor type

There are three kinds of accessor: regular, _vector_ and _global_. Regular
accessors, or simply *accessors*, indicate an access to a
regular [Data Field](data field) of a grid. Vector accessors are used when accessing
an [Expandable Parameters](expandable parameters data field),
 *global accessors* indicates that the
data to be referred does not participate in the iteration and always
_point_ to a same *read only* datum to be used in the operator.

An *accessor type* is a `using` statement with this form

```c++
using name = accessor<I, intent, [location_type,] extent, N>;
```
or
```c++
using name = global_accessor<I>;
```

- `name` is the name associated to the accessor and will be used in the
  implementation of the stencil operator. *Note:* _accessors names are
  technically optional, since their types can be substituted in all
  occurrences of their names. It is anyway part of the GridTools syntax,
  since a version not using them would be largely unreadable and very
  difficult to manage._

- `I` is an integer index. The indices of the accessors in a given
  stencil operartors *must* be ranging from 0 to N-1, where N is the
  number of accessors used by the stencil operator. No index can be
  replicated. If these rules are not followed the compilation
  fails. This is the last argument provided to global accessors.

- `intent` indicates the type of access the stencil operator makes to
  the data associated to the accessor. Possible extents are
  -- `enumtype::in` to specify *read-only* access
  -- `enumtyoe::inout` to specify *read-write* access. The `extent`
  for `inout`must be made of all zeros (see next points)

- `location_type` indicate in which location the accessor is assumed
  to access data. This is only needed when using *irregular grids* and
  cannot be specified for regular grids. Reference to
  [Irregular Grids](irregular grids documentation) for further
  details.

- `extent` defines the maximum offsets at which the implementation
  will access data around the point of evaluation. Extents are
  templates that takes a list of pairs of integer numbers. Every pair
  identify a dimension of the iteration space. The first number (<=0)
  indicates the offset in the direction of *decreasing* indices (also
  called *minus direction*), while the second (>=0) indicates the
  offset in the direction of *increasing* indices (also called *plus
  direction*). For example `extent< -1,1, 0,2, -2,1 >` specifies an
  access of one element in the direction of decreasing indices (-1)
  and one in the direction of increasing indices (+1) in the first
  dimension; two elements in the plus direction (+2) and no elements
  in the minus direction (0) in the second dimension, and finally two
  elements in the minus direction and one in the plus for the third
  dimension. All the numbers are *defaulted to 0*, so that `extent<>`is
  a valid extent. `extent<>` is also the default extent of an accessor
  and can be omitted if the last template argument takes also the
  default value (see next point). *Note:* _An extent with smaller
  offsets that the ones the implementation is using will result in a
  runtime error, while extents bigger that the one actually accessed
  by the implementation will result in performance loss._

- `N` identifies the number of dimensions of the [Data Field](data
  field). By default this value is set to 3. *Note:* _See
  [Advanced Access Specification] For more techniques to access data
  into a data field, especially of there are more that 3 dimensions._


### arg_list
The `arg_list` is a `using` statement like the following

```c++
using arg_list = accessor_list< _accessors_ >;
```

where `_accessors_` is a comma separated list of all the accessors
specified before. *Note:* _this is necessary since C++ cannot infer
what types have been defined as accessors._

### Do method 

The `Do` methods takes at most two(2) arguments, the
  type of the first one is the template type and it is usually called
  `Eval`. The second argument of the `Do` is a *vertical region*,
  discussed in [Splitter and Axis]. Multiple versions of the `Do` can
  be defined in the same stencil operator with different vertical
  region. *Note:* _there cannot be a _gaps_ in the vertical regions of
  a given stencil operator. If an operator is not defined in a given
  region, which is in the middle of other regions used, then it must
  be defined anyway and left empty._

  A `Do` method can be defined with a single template argument, so to
  skip the vertical region. In this case there could be *only one*
  `Do` method implementation in the stencil operator that will be
  called for each point of the iteration space.

  The return statement of the `Do` method is usually `void`. In case
  of reduction a `Do` method should return a value that can be used in
  the reduction (i.e., the value returned should be convertible to the
  arguments of the reduction operator).

## Example

```c++
    struct flx_function {

        using out = accessor< 0, enumtype::inout >;
        using in  = accessor< 1, enumtype::in, extent< 0, 1, 0, 0 > >;
        using lap = accessor< 2, enumtype::in, extent< 0, 1, 0, 0 > >;

        using arg_list = accessor_list< out, in, lap > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_flx) {
            eval(out()) = eval(lap(1, 0, 0)) - eval(lap(0, 0, 0));
            if (eval(out()) * (eval(in(1, 0, 0)) - eval(in(0, 0, 0))) > 0) {
                eval(out()) = 0.;
            }
        }
    };
```

## Implementation

The implementation is specified in the body of the `Do` methods. The
stencil operators can have other member functions that can be called
from the `Do`
methods. [The interface of the `Do` methods](#do-method) has been
already discussed in the previous section.  In this Section we
describe [how data can be accessed](#using-eval-for-regular-grids) and
[how other operators can be called](#calling-other-operators) from
within the `Do`'s body to perform a computation.

### Using eval For Regular Grids
Let us assume the `Do` methods has the following signature:

```c++
template <typename Eval>
GT_FUNCTION static
void Do(Eval const& eval, region);
```

The way to access data corresponding to a certain data field passed to
it, is to indicate the corresponding accessor as argument to the
`eval` argument, as follow:

```c++
eval(accessor_name())
```

*Note:* _The parentheses after `accessor_name` indicate the default constructor
of the accessor. This is a technicality necessary to make the
syntax legal in C++_

The previous syntax is said to *evaluate the accessor at the
evaluation point*.

For [Regular Grids] values can be accessed at offsets (relative to the
evaluation point) passing to the constructor
of the accessor a sequence of integer indices, as follows:

```c++
eval(accessor_name(1,0,-1))
```

This means to access an element at an offset of 1 in the first
dimension (plus direction) of the iteration space, and an offset of 1 in the minus direction
in the third dimension. A way to think of it is to consider the point
of evaluation as a triplet `i`, `j` and `k`, and those offsets are
added to the current index coordinates to identifying the actual value
to access.

The evaluation returns a reference to the value for accessors with
`inout` intent, and a value for accessors with `in` intent. See
[Intent section for more information on the intents](#accessor-type).

The next example takes the difference between two value in the first
dimension and assign it to the output field:

```c++
eval(out()) = eval(in()) - eval(in(1,0,0));
```

When using expressions, the previous example can be simplified to
read:

```c++
eval(out()) = eval( in() -in(1,0,0) );
```

To use expressions and other more advanced techniques to access data
and specifying offsets refer to [Advanced Access Specification].

### Using eval For Irregular Grids

### Calling other operators
