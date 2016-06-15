/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once

#include "level.hpp"

namespace gridtools {
    // forward declaration
    template < typename T >
    struct run_functor;

    /**forward declaration*/
    template < typename PointerType, typename MetaType, ushort_t Dim >
    struct base_storage;

    /**forward declaration*/
    template < typename U >
    struct storage;

    /**forward declaration*/
    template < enumtype::platform T >
    struct backend_traits_from_id;

    /**
       @brief wasted code because of the lack of constexpr
       its specializations, given the backend subclass of \ref gridtools::_impl::run_functor, returns the corresponding
       enum of type \ref gridtools::_impl::BACKEND .
    */
    template < class RunFunctor >
    struct backend_type;

    /** @brief functor struct whose specializations are responsible of running the kernel
        The kernel contains the computational intensive loops on the backend. The fact that it is a functor (and not a
       templated method) allows for partial specialization (e.g. two backends may share the same strategy)
    */
    template < typename Backend >
    struct execute_kernel_functor;
}
