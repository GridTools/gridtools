/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "level.hpp"

namespace gridtools {
    // forward declaration
    template <typename T>
    struct run_functor;

    /**forward declaration*/
    template <typename PointerType, typename MetaType, ushort_t Dim>
    struct base_storage;

    /**forward declaration*/
    template <typename U>
    struct storage;

    /**forward declaration*/
    template <typename T>
    struct backend_traits_from_id;

    /**
       @brief wasted code because of the lack of constexpr
    */
    template <class RunFunctor>
    struct backend_type;

    /** @brief functor struct whose specializations are responsible of running the kernel
        The kernel contains the computational intensive loops on the backend. The fact that it is a functor (and not a
       templated method) allows for partial specialization (e.g. two backends may share the same strategy)
    */
    template <typename Backend>
    struct execute_kernel_functor;
} // namespace gridtools
