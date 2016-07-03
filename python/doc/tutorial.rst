.. _tutorial:

Tutorial
========

Implementing the Game of Life
-----------------------------

Let's start with a simple MATLAB script that implements the Game of Life on
multiple domains simultaneously.
The code is based on Chapter 12 of Cleve Moler's ebook `Experiments with
MATLAB <http://ch.mathworks.com/moler/exm/>`_. The book is freely available, as
are PDFs of single chapters, so please refer to them should you need a brief
introduction to the theory behind the Game of Life.
The code is as follows:

.. code-block:: MATLAB

    % Generate a random initial population
    X = sparse(50,50,4);
    X(19:32,19:32,:) = (rand(14,14,4) > .75);
    p0 = nnz(X);

    % Whether cells stay alive, die, or generate new cells depends
    % upon how many of their eight possible neighbors are alive.
    % Index vectors increase or decrease the centered index by one.
    n = size(X,1);
    p = [1 1:n-1];
    q = [2:n n];

    % Loop over 100 generations.
    for t = 1:100

       spy(X(:,:,1))
       title(num2str(t))
       drawnow

       % Count how many of the eight neighbors are alive.
       Y = X(:,p,:) + X(:,q,:) + X(p,:,:) + X(q,:,:) + ...
           X(p,p,:) + X(q,q,:) + X(p,q,:) + X(q,p,:);

       % A live cell with two live neighbors, or any cell with
       % three live neighbors, is alive at the next step.
       X = (X & (Y == 2)) | (Y == 3);
    end

.. note::
    This and other MATLAB examples, along with their Python counterparts can be
    found in the ``${GRIDTOOLS-ROOT}/python/samples`` directory.

A possible Python implementation using a function could be:

.. code-block:: python

    def game_of_life (X):
        # Create index lists
        n = X.shape[0]
        q = np.r_[0, 0:n-1]
        p = np.r_[1:n n-1]

        Y = X[:,p,:] + X[:,q,:] + X[p,:,:] + X]q,:,:] + \
            X[p,p,:] + X[q,q,:] + X[p,q,:] + X]q,p,:]

        X = (X and (Y == 2)) or (Y == 3)

        return X

    # Generate a random initial population
    domain = (50, 50, 4)
    X = np.zeros(domain)
    X[19:33,19:33,:] = np.random.rand(14,14,4) > .75

    # Loop over 100 iterations
    for i in range(100):
        X = game_of_life(X)

Python and Numpy's slicing syntax makes it relatively easy to replicate MATLAB's
vectorized computation. Notice the slightly different definitions of index
vectors, due to Python's own slicing and array indexing rules.


Stencils and halos
------------------

Now, let's introduce elements from the Gridtools4Py package. A *stencil* can be
defined as an iterative operator or function that updates array elements
according to some fixed pattern. Such pattern usually accesses elements in the
neighborhood of the element to update; we call *halo* the maximum extent of the
pattern. The Game of Life we just implemented is a clear example of a stencil
computation that has an halo of 1 element in all planar directions: each element
of coordinates ``(i,j)`` needs data from elements ``(i+1,j), (i,j+1), (i-1,j),
(i,j-1), (i+1,j+1), (i-1,j-1), (i+1,j-1), (i-1,j+1)``. For reference, this kind
of pattern is also called `Moore neighborhood
<https://en.wikipedia.org/wiki/Moore_neighborhood>`_.

A stencil can only be applied to elements that have a number of neighbors at
least equal to the stencil's halo in each direction. Thus, the elements on the
domain boundaries have to be excluded from the operation, that can only be
performed on the *interior points* of the domain.

Gridtools, and consequently Gridtools4Py, do not use an halo on the vertical
direction (usually identified with the ``k`` index); as such, halos are only
defined along planar directions ``i`` and ``j``.

Creating a stencil
------------------

The simplest way to implement our Game of Life with Gridtools4Py is to use the
:func:`gridtools.stencil.Stencil.kernel` decorator on the ``game_of_life``
function. In Gridtools4Py's terminology, a *kernel* or *entry point function* is
the function that specifies the operations that will be performed by the
stencil. The decorator will transform our function into a callable object
subclassing :class:`Stencil`; however, there are no changes exposed to the end
user: ``game_of_life`` can be invoked as if it still was a function. When run,
Gridtools4Py's internal machinery will analyze the stencil code and verify that
it is fully compliant with the DSL syntax.

Since Gridtools4Py aims to mantain compatibility with Gridtools' C++ API, some
additional modifications to our initial Python Game of Life are required: the
vectorized syntax is not supported, so we have to explicitly indicate all the
neighbors we need to access, and we have to iterate only on the interior of the
domain, excluding the halo points.

Given a 3D array, the :func:`gridtools.stencil.Stencil.get_interior_points`
function returns a generator that yields tuples with the coordinates of interior
points for the domain, taking the halo into account. The array is traversed
first along the ``k`` direction, either bottom-up or top-down. Halo and k
direction can be set through the :func:`gridtools.stencil.Stencil.set_halo` and
:func:`gridtools.stencil.Stencil.set_k_direction` functions.

Gridtools has quite strict rules regarding assignment to arrays with
data coming from the same arrays. In our case, it is necessary to split input
and output data into two different arrays. For more details about this, please
refer to https://github.com/eth-cscs/gridtools/wiki/Data-Dependencies-Analysis-in-GridTools

Finally, stencil kernels must return ``None`` (another reason to have a separate
array for output data).

Our Game of Life can thus be rewritten as follows:

.. code-block:: python

    from gridtools.stencil import Stencil

    @Stencil.kernel
    def game_of_life (out_X, in_X):
        """
        Game of life implemented as a procedural stencil
        """
        for p in Stencil.get_interior_points (out_X):

            Y = in_X[p + (1,0,0)]  + in_X[p + (1,1,0)]   + \
                in_X[p + (0,1,0)]  + in_X[p + (-1,1,0)]  + \
                in_X[p + (-1,0,0)] + in_X[p + (-1,-1,0)] + \
                in_X[p + (0,-1,0)] + in_X[p + (1,-1,0)]

            out_X[p] = (in_X[p] and (Y == 2)) or (Y == 3)


    # Generate a random initial population
    domain = (50, 50, 4)
    in_X = np.zeros(domain)
    in_X[19:33,19:33,:] = np.random.rand(14,14,4) > .75
    out_X = np.copy(in_X)

    # Set halo and k direction for the stencil
    Stencil.set_halo( (1,1,1,1) )
    Stencil.set_k_direction('forward')

    # Loop over 100 iterations
    for i in range(100):
        game_of_life(out_X = out_X, in_X = in_X)
        in_X = out_X


Running with different backends
-------------------------------

If no errors are reported, the stencil can be executed with any of the backends
provided: Python, C++ or CUDA. Each backend has its pros and cons:

Python
    Can be used to quickly prototype stencils, catch runtime errors and verify
    results, as it doesn't require any compilation time. However, it is the
    backend with the lowest performance.
C++
    The stencil code is automatically translated into C++ using the Gridtools
    API and then compiled. Runs much faster than Python, but results and runtime
    errors can be more difficult to diagnose given the heavily templated nature
    of Gridtools' C++ code.
CUDA
    Translates the code to C++ and attempts to run it using Gridtools' own CUDA
    backend, provided that the system has a CUDA-capable GPU. Provides even
    faster processing than the C++ backend.

The default backend is Python. To select a different one, use the function
:func:`gridtools.stencil.Stencil.set_backend` with a string argument
corresponding to the desired backend:

.. code-block:: python

    from gridtools.stencil import Stencil

    @Stencil.kernel
    def game_of_life (out_X, in_X):
        """
        Game of life implemented as a procedural stencil
        """
        for p in Stencil.get_interior_points (out_X):

            Y = in_X[p + (1,0,0)]  + in_X[p + (1,1,0)]   + \
                in_X[p + (0,1,0)]  + in_X[p + (-1,1,0)]  + \
                in_X[p + (-1,0,0)] + in_X[p + (-1,-1,0)] + \
                in_X[p + (0,-1,0)] + in_X[p + (1,-1,0)]

            out_X[p] = (in_X[p] and (Y == 2)) or (Y == 3)


    # Generate a random initial population
    domain = (50, 50, 4)
    in_X = np.zeros(domain)
    in_X[19:33,19:33,:] = np.random.rand(14,14,4) > .75
    out_X = np.copy(in_X)

    # Set halo, k direction and backend for the stencil
    Stencil.set_halo( (1,1,1,1) )
    Stencil.set_k_direction('forward')
    Stencil.set_backend('c++')

    # Loop over 100 iterations
    for i in range(100):
        game_of_life(out_X = out_X, in_X = in_X)
        in_X = out_X


Procedural stencils and Object-Oriented stencils
------------------------------------------------

We call *procedural stencil* a stencil created through decoration of a simple
function, because these are intended to enable the creation of scripts and
programs following a procedural programming model (similar to what is
usually done in MATLAB).

It is also possible to create *object-oriented stencils*, though it requires a
little extra effort, that will be detailed in the next section.

Both procedural and object-oriented stencils have their pros and cons:

Procedural:
    *   Quick and easy to create by decorating a function. Do not require any
        knowledge of object-oriented programming;
    *   Always use the values for halo, k direction and backend that were set
        globally through :func:`gridtools.stencil.Stencil.set_halo`,
        :func:`gridtools.stencil.Stencil.set_k_direction` and
        :func:`gridtools.stencil.Stencil.set_backend`;
    *   Does not support the creation of stages outside the kernel function.
    
        .. note::
            This limitation may be removed with future developments.

Object-oriented:
    *   Slightly more complex to define; require knowledge of object-oriented
        programming concepts;
    *   Each object-oriented stencil can use an individual setting for halo,
        k direction adn backend;
    *   Can declare and use *temporary arrays* (a storage type provided by
        Gridtools) that simplifies the creation of complex stencils and may
        offer slightly better performance;
    *   Can create stages outside the kernel function by defining additional
        methods.

In brief, procedural stencils are easier to use, while object-oriented stencils
are more powerful and offer more features.


Creating a stencil with OOP
---------------------------

To create an object-oriented stencil we must first create a class that inherits
:class:`MultiStageStencil`, and override the default constructor:

.. code-block:: python

    from gridtools.stencil import Stencil, MultiStageStencil

    class GameOfLife (MultiStageStencil):
        def __init__ (self):
            super ( ).__init__ ( )

Our stencil obviously needs the entry point function (or kernel). We add that by
defining a method and decorating it with the usual
:func:`gridtools.stencil.Stencil.kernel` decorator:

.. code-block:: python

    from gridtools.stencil import Stencil, MultiStageStencil

    class GameOfLife (MultiStageStencil):
        """
        Game of life implemented as an object-oriented stencil
        """
        def __init__ (self):
            super ( ).__init__ ( )

        @Stencil.kernel
        def kernel (self, out_X, in_X):
            for p in self.get_interior_points (out_X):

                Y = in_X[p + (1,0,0)]  + in_X[p + (1,1,0)]   + \
                    in_X[p + (0,1,0)]  + in_X[p + (-1,1,0)]  + \
                    in_X[p + (-1,0,0)] + in_X[p + (-1,-1,0)] + \
                    in_X[p + (0,-1,0)] + in_X[p + (1,-1,0)]

                out_X[p] = (in_X[p] and (Y == 2)) or (Y == 3)

The method itself can have any name, just be sure to use the decorator to
identify it as the kernel. Also note that to iterate over the interior points of
the domain we are now using the
:func:`gridtools.stencil.MultiStageStencil.get_interior_points` method, called
as ``self.get_interior_points``, instead of the static function
``Stencil.get_interior_points``. This allows the object-oriented stencil to use
its own individual settings for halo and k direction.

Now that we have defined our custom stencil class, we can create a stencil object
by instancing the class. Halo, k direction and backend for the stencil can be set
using methods with the same name of the ``Stencil`` setter functions that we used
earlier:

.. code-block:: python

    # Initialize stencil
    stencil = GameOfLife()
    stencil.set_halo( (1,1,1,1) )
    stencil.set_k_direction('forward')
    stencil.set_backend('c++')

To execute the stencil, we use the :func:`gridtools.stencil.MultiStageStencil.run`
method. Here is the complete Game of Life example with an object-oriented stencil:

.. code-block:: python

    from gridtools.stencil import Stencil, MultiStageStencil

    class GameOfLife (MultiStageStencil):
        """
        Game of life implemented as an object-oriented stencil
        """
        def __init__ (self):
            super ( ).__init__ ( )

        @Stencil.kernel
        def kernel (self, out_X, in_X):
            for p in self.get_interior_points (out_X):

                Y = in_X[p + (1,0,0)]  + in_X[p + (1,1,0)]   + \
                    in_X[p + (0,1,0)]  + in_X[p + (-1,1,0)]  + \
                    in_X[p + (-1,0,0)] + in_X[p + (-1,-1,0)] + \
                    in_X[p + (0,-1,0)] + in_X[p + (1,-1,0)]

                out_X[p] = (in_X[p] and (Y == 2)) or (Y == 3)


    # Generate a random initial population
    domain = (50, 50, 4)
    in_X = np.zeros(domain)
    in_X[19:33,19:33,:] = np.random.rand(14,14,4) > .75
    out_X = np.copy(in_X)

    # Initialize stencil
    stencil = GameOfLife()
    stencil.set_halo( (1,1,1,1) )
    stencil.set_k_direction('forward')
    stencil.set_backend('c++')

    # Loop over 100 iterations
    for i in range(100):
        stencil.run(out_X = out_X, in_X = in_X)
        in_X = out_X
