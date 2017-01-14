/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#pragma once
#include "../iterate_domain.hpp"

/** @file iterate_domain for expandable parameters*/

namespace gridtools {

    template < typename T >
    struct is_iterate_domain;

    /**
       @brief iterate_domain specific for when expandable parameters are used

       In expandable parameter computations the user function is repeated a specific amount of time in
       each stencil. The parameters are stored in a storage list, and consecutive elements of the list
       are accessed in each user function.
       This struct "decorates" the base iterate_domain instance with a static const integer ID, which
       records the current position in the storage list, and reimplements the operator() in order to
       access the storage list at the correct offset.

       \tparam IterateDomain base iterate_domain class. Might be e.g. iterate_domain_host or iterate_domain_cuda
       \tparam Position the current position in the expandable parameters list
     */
    template < typename IterateDomain, ushort_t Position >
    struct iterate_domain_expandable_parameters : public IterateDomain {

        GRIDTOOLS_STATIC_ASSERT(is_iterate_domain< IterateDomain >::value, "wrong type");
        static const ushort_t ID = Position - 1;
        typedef IterateDomain super;
        typedef IterateDomain iterate_domain_t;

#ifdef CXX11_ENABLED
        // user protections
        template < typename... T >
        GT_FUNCTION iterate_domain_expandable_parameters(T const &... other_)
            : super(other_...) {
            GRIDTOOLS_STATIC_ASSERT((sizeof...(T) == 1), "The eval() is called with the wrong arguments");
        }
#endif

        template < typename T, ushort_t Val >
        GT_FUNCTION iterate_domain_expandable_parameters(iterate_domain_expandable_parameters< T, Val > const &other_)
            : super(other_) {
            GRIDTOOLS_STATIC_ASSERT((sizeof(T)),
                "The \'eval\' argument to the Do() method gets copied somewhere! You have to pass it by reference.");
        }

        using super::operator();

        /**
       @brief set the offset in the storage_list and forward to the base class

       when the vector_accessor is passed to the iterate_domain we know we are accessing an
       expandable parameters list. Accepts rvalue arguments (accessors constructed in-place)

       \param arg the vector accessor
     */
        // rvalue
        template < uint_t ACC_ID, enumtype::intend Intent, typename Extent, uint_t Size >
        GT_FUNCTION typename super::iterate_domain_t::template accessor_return_type<
            accessor< ACC_ID, Intent, Extent, Size > >::type
        operator()(vector_accessor< ACC_ID, Intent, Extent, Size > const &arg) {
            typedef typename super::template accessor_return_type< accessor< ACC_ID, Intent, Extent, Size > >::type
                return_t;
// check that if the storage is written the accessor is inout

#ifdef CUDA8
            GRIDTOOLS_STATIC_ASSERT(is_extent< Extent >::value, "wrong type");
            const typename alias< accessor< ACC_ID, Intent, Extent, Size >, dimension< Size - 1 > >::template set< ID >
                tmp_(arg.offsets());
#else
            accessor< ACC_ID, Intent, Extent, Size > tmp_(arg);
            tmp_.template set< 1 >(ID);
#endif
            return super::operator()(tmp_);
        }
    };

    template < typename T >
    struct is_iterate_domain_expandable_parameters : boost::mpl::false_ {};

    template < typename T, ushort_t Val >
    struct is_iterate_domain_expandable_parameters< iterate_domain_expandable_parameters< T, Val > >
        : boost::mpl::true_ {};

    template < typename T, ushort_t Val >
    struct is_iterate_domain< iterate_domain_expandable_parameters< T, Val > > : boost::mpl::true_ {};
}
