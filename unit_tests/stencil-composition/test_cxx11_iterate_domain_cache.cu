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
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include "common/defs.hpp"
#include "stencil-composition/empty_extent.hpp"
#include "stencil-composition/backend.hpp"
#include "stencil-composition/caches/cache_metafunctions.hpp"
#include "stencil-composition/interval.hpp"
#include "stencil-composition/stencil-composition.hpp"
#include "common/generic_metafunctions/fusion_map_to_mpl_map.hpp"
#include "stencil-composition/caches/extract_extent_caches.hpp"

using namespace gridtools;
using namespace enumtype;

// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval< level< 0, -1 >, level< 2, 1 > > axis;
typedef gridtools::interval< level< 0, -1 >, level< 0, -1 > > kminimum;
typedef gridtools::interval< level< 0, 1 >, level< 1, -1 > > krange1;
typedef gridtools::interval< level< 1, 1 >, level< 2, -2 > > krange2;
typedef gridtools::interval< level< 2, -1 >, level< 2, -1 > > kmaximum;

typedef layout_map< 0, 1 > layout_ij_t;
typedef backend< Cuda, GRIDBACKEND, Block >::storage_type< float_type,
    backend< Cuda, GRIDBACKEND, Block >::storage_info< 0, layout_ij_t > >::type storage_type;

typedef arg< 0, storage_type > p_in1;
typedef arg< 1, storage_type > p_in2;
typedef arg< 2, storage_type > p_in3;
typedef arg< 3, storage_type > p_in4;
typedef arg< 4, storage_type > p_out;

struct functor1 {
    typedef accessor< 0, enumtype::in, extent< 0, 0, 0, 0, -1, 0 > > in1;
    typedef accessor< 1, enumtype::in, extent< 0, 0, 0, 0, -1, 0 > > in3;
    typedef accessor< 2, enumtype::in, extent< 0, 0, 0, 0, -1, 0 > > in4;

    typedef accessor< 3, enumtype::inout, extent< 0, 0, 0, 0, 0, 1 > > out;
    typedef boost::mpl::vector< in1, in3, in4, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, kminimum) {}
    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, krange1) {}
    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, krange2) {}
};

struct functor2 {
    typedef accessor< 0, enumtype::in, extent< 0, 0, 0, 0, -1, 0 > > in1;
    typedef accessor< 1, enumtype::in, extent< 0, 0, 0, 0, -1, 0 > > in2;
    typedef accessor< 2, enumtype::in, extent< 0, 0, 0, 0, -1, 0 > > in4;

    typedef accessor< 3, enumtype::inout, extent< 0, 0, 0, 0, 0, 1 > > out;
    typedef boost::mpl::vector< in1, in2, in4, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, kmaximum) {}
    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, krange2) {}
};

typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, -1 > > kmin_and_range1;
typedef gridtools::interval< gridtools::level< 1, 1 >, gridtools::level< 2, -1 > > krange2_and_max;
typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 2, -1 > > kall;

typedef detail::cache_impl< K, p_in1, flush, kminimum > cache1_t;
typedef detail::cache_impl< K, p_in2, flush, kmin_and_range1 > cache2_t;
typedef detail::cache_impl< K, p_in3, flush, krange2_and_max > cache3_t;
typedef detail::cache_impl< K, p_in4, local, kmaximum > cache4_t;
typedef detail::cache_impl< K, p_out, flush, kall > cache5_t;

typedef boost::mpl::vector5< cache1_t, cache2_t, cache3_t, cache4_t, cache5_t > caches_t;

typedef decltype(gridtools::make_stage< functor1 >(p_in1(), p_in3(), p_in4(), p_out())) esf1k_t;
typedef decltype(gridtools::make_stage< functor2 >(p_in1(), p_in2(), p_in4(), p_out())) esf2k_t;

typedef boost::mpl::vector2< esf1k_t, esf2k_t > esfk_sequence_t;

TEST(iterate_domain_cache, forward_backward) {

    typedef metadata_set< boost::mpl::vector1< pointer< storage_type::storage_info_type > > > metadata_vector_t;
    typedef boost::mpl::vector5< pointer< storage_type >,
        pointer< storage_type >,
        pointer< storage_type >,
        pointer< storage_type >,
        pointer< storage_type > > storages_t;
    typedef boost::fusion::result_of::as_vector< storages_t >::type storages_tuple_t;
    typedef boost::mpl::vector5< p_in1, p_in2, p_in3, p_in4, p_out > esf_args_t;
    typedef local_domain< storages_tuple_t, metadata_vector_t, esf_args_t, false > local_domain_t;

    typedef boost::mpl::vector2< extent< -1, 2, -2, 1 >, extent< -2, 1, -3, 2 > > extents_t;

    typedef typename boost::mpl::fold< extents_t,
        extent< 0, 0, 0, 0 >,
        enclosing_extent< boost::mpl::_1, boost::mpl::_2 > >::type max_extent_t;

    typedef iterate_domain_arguments< backend_ids< Cuda, GRIDBACKEND, Block >,
        local_domain_t,
        esfk_sequence_t,
        extents_t,
        max_extent_t,
        caches_t,
        block_size< 32, 4, 1 >,
        block_size< 32, 4, 1 >,
        gridtools::grid< axis >,
        boost::mpl::false_,
        notype > iterate_domain_arguments_t;

    using iterate_domain_cache_t = iterate_domain_cache< iterate_domain_arguments_t >;

    using k_flushing_caches_indexes_t = iterate_domain_cache_t::k_flushing_caches_indexes_t;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal< k_flushing_caches_indexes_t,
            boost::mpl::vector4< static_uint< 0 >, static_uint< 1 >, static_uint< 2 >, static_uint< 4 > > >::value),
        "Error");

    using iteration_policy1_t =
        _impl::iteration_policy< kminimum::FromLevel, kminimum::ToLevel, static_uint< 2 >, enumtype::forward >;

    using final_flush_indexes1_t = iterate_domain_cache_t::kcache_final_flush_indexes< iteration_policy1_t >::type;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal< final_flush_indexes1_t, boost::mpl::vector1< static_uint< 0 > > >::value), "Error");

    using iteration_policy2_t =
        _impl::iteration_policy< krange1::FromLevel, krange1::ToLevel, static_uint< 2 >, enumtype::forward >;

    using final_flush_indexes2_t = iterate_domain_cache_t::kcache_final_flush_indexes< iteration_policy2_t >::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal< final_flush_indexes2_t, boost::mpl::vector1< static_uint< 1 > > >::value), "Error");

    using iteration_policy3_t =
        _impl::iteration_policy< krange2::FromLevel, krange2::ToLevel, static_uint< 2 >, enumtype::forward >;

    using final_flush_indexes3_t = iterate_domain_cache_t::kcache_final_flush_indexes< iteration_policy3_t >::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< final_flush_indexes3_t >::value == 0), "Error");

    using iteration_policy4_t =
        _impl::iteration_policy< kmaximum::FromLevel, kmaximum::ToLevel, static_uint< 2 >, enumtype::forward >;

    using final_flush_indexes4_t = iterate_domain_cache_t::kcache_final_flush_indexes< iteration_policy4_t >::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal< final_flush_indexes4_t, boost::mpl::vector2< static_uint< 2 >, static_uint< 4 > > >::value),
        "Error");
    // backward
    using iteration_policy5_t =
        _impl::iteration_policy< kmaximum::FromLevel, kmaximum::ToLevel, static_uint< 2 >, enumtype::backward >;

    using final_flush_indexes5_t = iterate_domain_cache_t::kcache_final_flush_indexes< iteration_policy5_t >::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< final_flush_indexes5_t >::value == 0), "Error");

    using iteration_policy6_t =
        _impl::iteration_policy< krange2::FromLevel, krange2::ToLevel, static_uint< 2 >, enumtype::backward >;

    using final_flush_indexes6_t = iterate_domain_cache_t::kcache_final_flush_indexes< iteration_policy6_t >::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal< final_flush_indexes6_t, boost::mpl::vector1< static_uint< 2 > > >::value), "Error");

    using iteration_policy7_t =
        _impl::iteration_policy< krange1::FromLevel, krange1::ToLevel, static_uint< 2 >, enumtype::backward >;

    using final_flush_indexes7_t = iterate_domain_cache_t::kcache_final_flush_indexes< iteration_policy7_t >::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< final_flush_indexes7_t >::value == 0), "Error");

    using iteration_policy8_t =
        _impl::iteration_policy< kminimum::FromLevel, kminimum::ToLevel, static_uint< 2 >, enumtype::backward >;

    using final_flush_indexes8_t = iterate_domain_cache_t::kcache_final_flush_indexes< iteration_policy8_t >::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal< final_flush_indexes8_t,
                                boost::mpl::vector3< static_uint< 0 >, static_uint< 1 >, static_uint< 4 > > >::value),
        "Error");
}
