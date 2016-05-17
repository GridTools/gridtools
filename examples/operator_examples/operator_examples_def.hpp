//
// Created by Xiaolin Guo on 20.04.16.
//

#pragma once

#include <gridtools.hpp>
#include <stencil-composition/stencil-composition.hpp>

namespace operator_examples {

    using namespace gridtools;
    using namespace enumtype;
    using namespace expressions;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

    using backend_t = BACKEND;
    using icosahedral_topology_t = ::gridtools::icosahedral_topology<backend_t>;

}

