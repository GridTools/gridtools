.. include:: ../defs.hrst

.. _introduction:

============
Introduction
============

-----------------
What Is GridTools
-----------------

The |GT| (GT) framework is a set of libraries and utilities to develop performance portable applications in which
stencil operations on grids are central. The focus of the project is on regular and block-structured grids as are
commonly found in the weather and climate application field. In this context, GT provides a useful level of abstraction
to enhance productivity and obtain excellent performance on a wide range of computer architectures. Additionally, it
addresses the challenges that arise from integration into production code, such as the expression of boundary
conditions, or conditional execution. The framework is structured such that it can be called from different weather
models (numerical weather and climate codes) or programming interfaces, and can target various computer architectures.
This is achieved by separating the GT core library in a user facing part (frontend) and architecture specific (backend)
parts. The core library also abstracts various possible data layouts and applies optimizations on stages with multiple
stencils. The core library is complemented by facilities to interoperate with other languages (such as C and Fortran),
to aid code development and a communication layer.

|GT| provides optimized backends for GPUs and manycore architectures. Stencils can be run efficiently on different
architectures without any code change needed. Stencils can be built up by small composeable units called stages, using
|GT| domain-specific language. Such a functor can be as simple as being just a copy stencil, copying data from one field
to another:

.. code-block:: gridtools

  struct copy_functor {
      using in = in_accessor<0>;
      using out = inout_accessor<1>;

      using param_list = make_param_list<in, out>;

      template <typename Evaluation>
      GT_FUNCTION static void apply(Evaluation eval) {
          eval(out()) = eval(in());
      }
  };

Several such stages can be composed into a computation and be applied on each grid-point of a grid. Requiring this
abstract descriptions of a stencils, the DSL allows |GT| can apply architecture-specific optimizations to the stencil
computations in order to be optimal on the target hardware.


^^^^^^^^^^^^
Dependencies
^^^^^^^^^^^^

|GT| requires at least a header-only installation of Boost_. Besides some boost utilities, it depends on ``boost::mpl`` and ``boost::fusion``.

Additionally, |GT| requires a recent version of CMake_.

.. _Boost: https://www.boost.org/
.. _CMake: https://www.cmake.org/

|GT| requires a modern compiler. A list of supported compilers can be found on `github <https://github.com/GridTools/gridtools>`_.


.. _installation:

--------------------
Installation and Use
--------------------

^^^^^^^^^^^^^
Simple Script
^^^^^^^^^^^^^

We first provide a sample of the commands needed to enable the multicore and CUDA backends, install them in ``/usr/local``,
and run the gridtools tests.

.. code-block:: shell

 git clone http://github.com/eth-cscs/gridtools.git
 cd gridtools
 mkdir build && cd build
 cmake -DGT_ENABLE_BACKEND_MC=ON -DGT_ENABLE_BACKEND_CUDA=ON -DCMAKE_INSTALL_PREFIX=/usr/local ..
 make install -j4
 make test

|GT| uses CMake as building system. The installation can be configured using `ccmake`. The following variables control the back-ends that will be supported by the runtime components of the installation, namely :ref:`GCL <halo-exchanges>`.

.. code-block:: shell

 GT_ENABLE_BACKEND_CUDA # For CUDA GPUs
 GT_ENABLE_BACKEND_X86  # For cache based multicores
 GT_ENABLE_BACKEND_NAIVE  # For naive implementation
 GT_ENABLE_BACKEND_MC   # For optimized multicores and KNL

All the targets can be installed and used at the same time, but some runtime components may lead to incompatibilities or complex environments to make the codes run. It may be more effective to do multiple installs of the library for different targets in this case.

At the moment the only runtime component of |GT| is the GCL, the communication module. By default, this component will be installed in the system as a single-node mockery of the full distributed memory capability. To enable the full GCL you must set to ``ON`` the following variable

.. code-block:: shell

 GT_USE_MPI

In addition, it may be convenient to install the library without compiling and running the tests, that use multiple nodes. To do so you can set to ``ON`` the following variable:

.. code-block:: shell

 GT_GCL_ONLY

Other variables can be useful for your system, and you can find their descriptions using ``ccmake``.

When installing |GT| all the source codes of the components will be copied to the installation path. To avoid compiling and running tests for specific components, |GT| allows to enable or disable components using the following variables, with obvious meaning:

.. code-block:: shell

 INSTALL_BOUNDARY_CONDITIONS
 INSTALL_DISTRIBUTED_BOUNDARIES
 INSTALL_COMMON
 INSTALL_STENCIL_COMPOSITION
 INSTALL_STORAGE
 INSTALL_C_BINDINGS
 INSTALL_INTERFACE
 INSTALL_TOOLS

To have access to these variables ``INSTALL_ALL`` should be set to ``OFF``.

Examples can be installed by enabling ``GT_INSTALL_EXAMPLES``. This will install the examples into the path given by
``GT_INSTALL_EXAMPLES_PATH``. The examples come with a standalone CMake project and can be built separately, e.g. with
the following set of instructions:

.. code-block:: shell

  cd examples
  mkdir build && cd build
  cmake ..
  make -j4

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Enabling Clang-CUDA and AMD GPU Support (Experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Besides NVCC-based compilation for NVIDIA GPUs, |GT| additionally supports Clang-based compilation for NVIDIA and AMD GPUs.
Due to a bug in Clang (https://bugs.llvm.org/show_bug.cgi?id=42061), this is only possible when not using OpenMP.
As all non-GPU backends of GridTools currently require OpenMP, they have to be disabled.
Then, Clang-based CUDA compilation can be activated by setting the CMake-variable ``GT_CUDA_COMPILATION_TYPE`` to ``Clang-CUDA``
and setting Clang as the main C++ compiler. A fully working command for configuring GridTools could look like this:

.. code-block:: shell

 export CXX=/path/to/clang++
 cmake -DGT_ENABLE_BACKEND_CUDA=ON \
     -DGT_CUDA_COMPILATION_TYPE=Clang-CUDA \
     -DGT_ENABLE_BACKEND_MC=OFF \
     -DGT_ENABLE_BACKEND_X86=OFF \
     -DGT_ENABLE_BACKEND_NAIVE=OFF \
     -DCMAKE_INSTALL_PREFIX=/usr/local \
     ..

Further, GridTools can also be compiled for AMD GPUs using AMD’s HIP. But due to limitations of AMD’s current HCC compiler,
the Clang-based HIP version has to be used, which is not yet officially distributed in the current HIP releases
(but will be the successor of HCC-based HIP). Compilation of a Clang-based HIP compiler is relatively straightforward and
documented in the `official HIP repository <https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md#hip-clang>`_.
Otherwise, the configuration is similar to the one for Clang-CUDA. Just set ``GT_CUDA_COMPILATION_TYPE`` to ``HIPCC_AMDGPU``
and the C++ compiler to HIPCC. For example:

.. code-block:: shell

 export CXX=/path/to/hipcc
 cmake -DGT_ENABLE_BACKEND_CUDA=ON \
     -DGT_CUDA_COMPILATION_TYPE=HIPCC-AMDGPU \
     -DGT_ENABLE_BACKEND_MC=OFF \
     -DGT_ENABLE_BACKEND_X86=OFF \
     -DGT_ENABLE_BACKEND_NAIVE=OFF \
     -DCMAKE_INSTALL_PREFIX=/usr/local \
     ..

.. note::

 The backend used for the AMD GPUs is also named “CUDA” for historical reasons as internally the same backend (with minor changes)
 is compiled for AMD and NVIDIA GPUs.


^^^^^^^^^^^^^^^
Using GridTools
^^^^^^^^^^^^^^^

Using |GT| follows standard CMake practices. To indicate where the |GT| can be found,
CMake should be provided with the variable ``GridTools_DIR``,
e.g. by calling CMake with ``-DGridTools_DIR=</path/to/gridtools/lib/cmake>``.
The ``CMakeLists.txt`` file should then contain the following lines:

.. code-block:: cmake

 find_package(GridTools VERSION ... REQUIRED)
 list(APPEND CMAKE_MODULE_PATH "${GridTools_MODULE_PATH}")

.. note::
 If GridTools uses the CUDA backend, you must call ``enable_language(CUDA)`` before finding the package.

Targets that need |GT| should link against ``GridTools::gridtools``. If the communication module is needed ``GridTools::gcl`` should be used instead.

.. code-block:: cmake

 add_library(my_library source.cpp)
 target_link_libraries(my_library PUBLIC GridTools::gridtools)

------------
Contributing
------------

Contributions to the |GT| set of libraries are welcome. However, our policy is that we will be the official maintainers and providers of the GridTools code. We believe that this will provide our users with a clear reference point for support and guarantees on timely interactions. For this reason, we require that external contributions to |GT| will be accepted after their authors provide to us a signed copy of a copyright release form to ETH Zurich.
