.. include:: ../defs.hrst

.. _introduction:

=====================
Introduction
=====================

-----------------------------
What Is GridTools
-----------------------------

The |GT| (GT) framework is a set of libraries and utilities to develop performance portable applications in which stencil operations on grids are central. The focus of the project is on regular and block-structured grids as are commonly found in the weather and climate application field. In this context, GT provides a useful level of abstraction to enhance productivity and obtain excellent performance on a wide range of computer architectures. Additionally, it addresses the challenges that arise from integration into production code, such as the expression of boundary conditions, or conditional execution. The framework is structured such that it can be called from different weather models (numerical weather and climate codes) or programming interfaces, and can target various computer architectures. This is achieved by separating the GT core library in a user facing part (frontend) and architecture specific (backend) parts. The core library also abstracts various possible data layouts and applies optimizations on stages with multiple stencils. The core library is complemented by facilities to interoperate with other languages (such as C and Fortran), to aid code development and a communication layer.

For a list of supported compilers refer to the `project Wiki on github <https://github.com/eth-cscs/gridtools/wiki/Supported-Compilers>`_.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^

|GT| requires at least a header-only installation of Boost_. Besides some boost utilities, it depends on ``boost::mpl`` and ``boost::fusion``.

Additionally, |GT| requires a recent version of CMake_.

.. _Boost: https://www.boost.org/
.. _CMake: https://www.cmake.org/

-----------------------------
Installation and Use
-----------------------------

|GT| uses CMake as building system. The installation can be configured using `ccmake`. The following variables control the back-ends that will be supported by the runtime components of the installation, namely :ref:`GCL <halo-exchanges>`.

.. code-block:: shell

 GT_ENABLE_TARGET_CUDA # For CUDA GPUs
 GT_ENABLE_TARGET_X86  # For cache based multicores and naive implementation
 GT_ENABLE_TARGET_MC   # For optimized multicores and KNL

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

.. todo:: Update to new examples ..

Additionally, examples can be compiled if ``GT_COMPILE_EXAMPLES`` is ``ON``. The examples can be installed if ``GT_INSTALL_EXAMPLES`` is ``ON``. The path where to install the examples is specified by ``GT_INSTALL_EXAMPLES_PATH`` and it is set to ``CMAKE_INSTALL_PREFIX`` by default.

^^^^^^^^^^^^^^^^^^^^
Simple Script
^^^^^^^^^^^^^^^^^^^^

Below a sample of the commands needed to enable the multicore and CUDA backends and install them in ``/usr/local``.

.. code-block:: shell

 git clone http://github.com/eth-cscs/gridtools.git
 cd gridtools
 mkdir build
 cd build
 cmake -DGT_ENABLE_TARGET_MC=ON -DGT_ENABLE_TARGET_CUDA=ON -DCMAKE_INSTALL_PREFIX=/usr/local ..
 make install
 make test

^^^^^^^^^^^^^^^^^^^^^
Using GridTools
^^^^^^^^^^^^^^^^^^^^^

Using |GT| follows standard CMake practices. To indicate where the |GT| can be found, CMake should be provided with the variable ``gridtools_DIR``, e.g. by calling CMake with ``-Dgridtools_DIR=</path/to/gridtools/lib/cmake>``. The ``CMakeLists.txt`` file should then contain the following lines:

.. code-block:: cmake

 find_package(GridTools VERSION ... REQUIRED)
 list(APPEND CMAKE_MODULE_PATH "${GridTools_MODULE_PATH}")

.. note::
 If GridTools uses the CUDA backend, you must call ``enable_language(CUDA)`` before finding the package.

Targets that need |GT| should link against ``GridTools::gridtools``. If the communication module is needed ``GridTools::gcl`` should be used instead.

.. code-block:: cmake

 add_library(my_library source.cpp)
 target_link_libraries(my_library PUBLIC GridTools::gridtools)

-----------------------------
Contributing
-----------------------------

Contributions to the |GT| set of libraries are welcome. However, our policy is that we will be the official maintainers and providers of the GridTools code. We believe that this will provide our users with a clear reference point for support and guarantees on timely interactions. For this reason, we require that external contributions to |GT| will be accepted after their authors provide to us a signed copy of a copyright release form to ETH Zurich.
