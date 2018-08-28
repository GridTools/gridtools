/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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
#include <tuple>
#include <type_traits>

#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/common/generic_metafunctions/fusion_map_to_mpl_map.hpp>
#include <gridtools/stencil-composition/backend.hpp>
#include <gridtools/stencil-composition/caches/cache_metafunctions.hpp>
#include <gridtools/stencil-composition/caches/extract_extent_caches.hpp>
#include <gridtools/stencil-composition/interval.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace enumtype;

// This is the definition of the special regions in the "vertical" direction
using axis_t = axis<2>::with_extra_offsets<1>;
using kminimum = axis_t::full_interval::first_level::shift<-1>;
using krange1 = axis_t::get_interval<0>;
using krange2 = axis_t::get_interval<1>::modify<0, -1>;
using kmaximum = axis_t::full_interval::last_level;

typedef storage_traits<platform::x86>::storage_info_t<0, 2> storage_info_ij_t;
typedef storage_traits<platform::x86>::data_store_t<float_type, storage_info_ij_t> storage_type;

typedef arg<0, storage_type> p_in1;
typedef arg<1, storage_type> p_in2;
typedef arg<2, storage_type> p_in3;
typedef arg<3, storage_type> p_in4;
typedef arg<4, storage_type> p_out;

struct functor1 {
    typedef accessor<0, enumtype::in, extent<0, 0, 0, 0, -1, 0>> in1;
    typedef accessor<1, enumtype::in, extent<0, 0, 0, 0, -1, 0>> in3;
    typedef accessor<2, enumtype::in, extent<0, 0, 0, 0, -1, 0>> in4;

    typedef accessor<3, enumtype::inout, extent<0, 0, 0, 0, 0, 1>> out;
    typedef boost::mpl::vector<in1, in3, in4, out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval, kminimum) {}
    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval, krange1) {}
    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval, krange2) {}
};

struct functor2 {
    typedef accessor<0, enumtype::in, extent<0, 0, 0, 0, -1, 0>> in1;
    typedef accessor<1, enumtype::in, extent<0, 0, 0, 0, -1, 0>> in2;
    typedef accessor<2, enumtype::in, extent<0, 0, 0, 0, -1, 0>> in4;

    typedef accessor<3, enumtype::inout, extent<0, 0, 0, 0, 0, 1>> out;
    typedef boost::mpl::vector<in1, in2, in4, out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval, kmaximum) {}
    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation &eval, krange2) {}
};

using kmin_and_range1 = krange1::modify<-1, 0>;
using krange2_and_max = krange2::modify<0, 1>;
using kall = axis_t::full_interval::modify<-1, 0>;

typedef decltype(gridtools::make_stage<functor1>(p_in1(), p_in3(), p_in4(), p_out())) esf1k_t;
typedef decltype(gridtools::make_stage<functor2>(p_in1(), p_in2(), p_in4(), p_out())) esf2k_t;

typedef boost::mpl::vector2<esf1k_t, esf2k_t> esfk_sequence_t;

TEST(iterate_domain_cache, flush) {

    typedef detail::cache_impl<K, p_in1, cache_io_policy::flush, kminimum> cache1_t;
    typedef detail::cache_impl<K, p_in2, cache_io_policy::flush, kmin_and_range1> cache2_t;
    typedef detail::cache_impl<K, p_in3, cache_io_policy::flush, krange2_and_max> cache3_t;
    typedef detail::cache_impl<K, p_in4, cache_io_policy::local, kmaximum> cache4_t;
    typedef detail::cache_impl<K, p_out, cache_io_policy::flush, kall> cache5_t;

    typedef boost::mpl::vector5<cache1_t, cache2_t, cache3_t, cache4_t, cache5_t> caches_t;

    typedef local_domain<std::tuple<p_in1, p_in2, p_in3, p_in4, p_out>, extent<>, false> local_domain_t;

    typedef boost::mpl::vector2<extent<-1, 2, -2, 1>, extent<-2, 1, -3, 2>> extents_t;

    typedef typename boost::mpl::
        fold<extents_t, extent<0, 0, 0, 0>, enclosing_extent_2<boost::mpl::_1, boost::mpl::_2>>::type max_extent_t;

    typedef iterate_domain_arguments<backend_ids<platform::cuda, GRIDBACKEND, strategy::block>,
        local_domain_t,
        esfk_sequence_t,
        extents_t,
        max_extent_t,
        caches_t,
        gridtools::grid<axis_t>,
        boost::mpl::false_,
        notype>
        iterate_domain_arguments_t;

    using iterate_domain_cache_t = iterate_domain_cache<iterate_domain_arguments_t>;

    using k_flushing_caches_indexes_t = iterate_domain_cache_t::k_flushing_caches_indexes_t;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<k_flushing_caches_indexes_t,
            boost::mpl::vector4<static_uint<0>, static_uint<1>, static_uint<2>, static_uint<4>>>::value),
        "Error");

    using iteration_policy1_t = iteration_policy<kminimum::FromLevel, kminimum::ToLevel, enumtype::forward>;

    using final_flush_indexes1_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy1_t>::type;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<final_flush_indexes1_t, boost::mpl::vector1<static_uint<0>>>::value), "Error");

    using iteration_policy2_t = iteration_policy<krange1::FromLevel, krange1::ToLevel, enumtype::forward>;

    using final_flush_indexes2_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy2_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<final_flush_indexes2_t, boost::mpl::vector1<static_uint<1>>>::value), "Error");

    using iteration_policy3_t = iteration_policy<krange2::FromLevel, krange2::ToLevel, enumtype::forward>;

    using final_flush_indexes3_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy3_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<final_flush_indexes3_t>::value == 0), "Error");

    using iteration_policy4_t = iteration_policy<kmaximum::FromLevel, kmaximum::ToLevel, enumtype::forward>;

    using final_flush_indexes4_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy4_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<final_flush_indexes4_t, boost::mpl::vector2<static_uint<2>, static_uint<4>>>::value),
        "Error");
    // backward
    using iteration_policy5_t = iteration_policy<kmaximum::ToLevel, kmaximum::FromLevel, enumtype::backward>;

    using final_flush_indexes5_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy5_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<final_flush_indexes5_t>::value == 0), "Error");

    using iteration_policy6_t = iteration_policy<krange2::ToLevel, krange2::FromLevel, enumtype::backward>;

    using final_flush_indexes6_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy6_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<final_flush_indexes6_t, boost::mpl::vector1<static_uint<2>>>::value), "Error");

    using iteration_policy7_t = iteration_policy<krange1::ToLevel, krange1::FromLevel, enumtype::backward>;

    using final_flush_indexes7_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy7_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<final_flush_indexes7_t>::value == 0), "Error");

    using iteration_policy8_t = iteration_policy<kminimum::ToLevel, kminimum::FromLevel, enumtype::backward>;

    using final_flush_indexes8_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy8_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal<final_flush_indexes8_t,
                                boost::mpl::vector3<static_uint<0>, static_uint<1>, static_uint<4>>>::value),
        "Error");
}

TEST(iterate_domain_cache, fill) {

    typedef detail::cache_impl<K, p_in1, cache_io_policy::fill, kminimum> cache1_t;
    typedef detail::cache_impl<K, p_in2, cache_io_policy::flush, kmin_and_range1> cache2_t;
    typedef detail::cache_impl<K, p_in3, cache_io_policy::fill, krange2_and_max> cache3_t;
    typedef detail::cache_impl<K, p_in4, cache_io_policy::local, kmaximum> cache4_t;
    typedef detail::cache_impl<K, p_out, cache_io_policy::flush, kall> cache5_t;

    typedef boost::mpl::vector5<cache1_t, cache2_t, cache3_t, cache4_t, cache5_t> caches_t;

    typedef std::tuple<p_in1, p_in2, p_in3, p_in4, p_out> esf_args_t;

    typedef local_domain<esf_args_t, extent<>, false> local_domain_t;

    typedef boost::mpl::vector2<extent<-1, 2, -2, 1>, extent<-2, 1, -3, 2>> extents_t;

    typedef typename boost::mpl::
        fold<extents_t, extent<0, 0, 0, 0>, enclosing_extent_2<boost::mpl::_1, boost::mpl::_2>>::type max_extent_t;

    typedef iterate_domain_arguments<backend_ids<platform::cuda, GRIDBACKEND, strategy::block>,
        local_domain_t,
        esfk_sequence_t,
        extents_t,
        max_extent_t,
        caches_t,
        gridtools::grid<axis_t>,
        boost::mpl::false_,
        notype>
        iterate_domain_arguments_t;

    using iterate_domain_cache_t = iterate_domain_cache<iterate_domain_arguments_t>;

    using k_filling_caches_indexes_t = iterate_domain_cache_t::k_filling_caches_indexes_t;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<k_filling_caches_indexes_t, boost::mpl::vector2<static_uint<0>, static_uint<2>>>::value),
        "Error");

    using iteration_policy1_t = iteration_policy<kminimum::FromLevel, kminimum::ToLevel, enumtype::forward>;

    using begin_fill_indexes1_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy1_t>::type;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<begin_fill_indexes1_t, boost::mpl::vector1<static_uint<0>>>::value), "Error");

    using iteration_policy2_t = iteration_policy<krange1::FromLevel, krange1::ToLevel, enumtype::forward>;

    using begin_fill_indexes2_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy2_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<begin_fill_indexes2_t>::value == 0), "Error");

    using iteration_policy3_t = iteration_policy<krange2::FromLevel, krange2::ToLevel, enumtype::forward>;

    using begin_fill_indexes3_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy3_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<begin_fill_indexes3_t, boost::mpl::vector1<static_uint<2>>>::value), "Error");

    using iteration_policy4_t = iteration_policy<kmaximum::FromLevel, kmaximum::ToLevel, enumtype::forward>;

    using begin_fill_indexes4_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy4_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<begin_fill_indexes4_t>::value == 0), "Error");

    // backward
    using iteration_policy5_t = iteration_policy<kmaximum::FromLevel, kmaximum::ToLevel, enumtype::backward>;

    using begin_fill_indexes5_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy5_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<begin_fill_indexes5_t, boost::mpl::vector1<static_uint<2>>>::value), "Error");

    using iteration_policy6_t = iteration_policy<krange2::FromLevel, krange2::ToLevel, enumtype::backward>;

    using begin_fill_indexes6_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy6_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<begin_fill_indexes6_t>::value == 0), "Error");

    using iteration_policy7_t = iteration_policy<krange1::FromLevel, krange1::ToLevel, enumtype::backward>;

    using begin_fill_indexes7_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy7_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<begin_fill_indexes7_t>::value == 0), "Error");

    using iteration_policy8_t = iteration_policy<kminimum::FromLevel, kminimum::ToLevel, enumtype::backward>;

    using begin_fill_indexes8_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy8_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<begin_fill_indexes8_t, boost::mpl::vector1<static_uint<0>>>::value), "Error");
}

TEST(iterate_domain_cache, epflush) {

    typedef std::tuple<p_in1, p_in2, p_in3, p_in4, p_out> esf_args_t;

    typedef local_domain<esf_args_t, extent<>, false> local_domain_t;

    typedef boost::mpl::vector2<extent<-1, 2, -2, 1>, extent<-2, 1, -3, 2>> extents_t;

    typedef typename boost::mpl::
        fold<extents_t, extent<0, 0, 0, 0>, enclosing_extent_2<boost::mpl::_1, boost::mpl::_2>>::type max_extent_t;

    typedef detail::cache_impl<K, p_in1, cache_io_policy::flush, kminimum> cachef1_t;
    typedef detail::cache_impl<K, p_in2, cache_io_policy::epflush, kmin_and_range1> cachef2_t;
    typedef detail::cache_impl<K, p_in3, cache_io_policy::epflush, krange2_and_max> cachef3_t;
    typedef detail::cache_impl<K, p_in4, cache_io_policy::epflush, kmaximum> cachef4_t;
    typedef detail::cache_impl<K, p_out, cache_io_policy::flush, kall> cachef5_t;

    typedef boost::mpl::vector5<cachef1_t, cachef2_t, cachef3_t, cachef4_t, cachef5_t> cachesf_t;

    typedef iterate_domain_arguments<backend_ids<platform::cuda, GRIDBACKEND, strategy::block>,
        local_domain_t,
        esfk_sequence_t,
        extents_t,
        max_extent_t,
        cachesf_t,
        gridtools::grid<axis_t>,
        boost::mpl::false_,
        notype>
        iterate_domain_arguments_t;

    using iterate_domain_cache_t = iterate_domain_cache<iterate_domain_arguments_t>;

    using k_flushing_caches_indexes_t = iterate_domain_cache_t::k_flushing_caches_indexes_t;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<k_flushing_caches_indexes_t, boost::mpl::vector2<static_uint<0>, static_uint<4>>>::value),
        "Error");

    using k_epflushing_caches_indexes_t = iterate_domain_cache_t::k_epflushing_caches_indexes_t;
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal<k_epflushing_caches_indexes_t,
                                boost::mpl::vector3<static_uint<1>, static_uint<2>, static_uint<3>>>::value),
        "Error");

    using iteration_policy1_t = iteration_policy<kminimum::FromLevel, kminimum::ToLevel, enumtype::forward>;

    using final_flush_indexes1_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy1_t>::type;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<final_flush_indexes1_t, boost::mpl::vector1<static_uint<0>>>::value), "Error");

    using iteration_policy2_t = iteration_policy<krange1::FromLevel, krange1::ToLevel, enumtype::forward>;

    using final_flush_indexes2_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy2_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<final_flush_indexes2_t, boost::mpl::vector1<static_uint<1>>>::value), "Error");

    using iteration_policy3_t = iteration_policy<krange2::FromLevel, krange2::ToLevel, enumtype::forward>;

    using final_flush_indexes3_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy3_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<final_flush_indexes3_t>::value == 0), "Error");

    using iteration_policy4_t = iteration_policy<kmaximum::FromLevel, kmaximum::ToLevel, enumtype::forward>;

    using final_flush_indexes4_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy4_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal<final_flush_indexes4_t,
                                boost::mpl::vector3<static_uint<2>, static_uint<3>, static_uint<4>>>::value),
        "Error");
    // backward
    using iteration_policy5_t = iteration_policy<kmaximum::ToLevel, kmaximum::FromLevel, enumtype::backward>;

    using final_flush_indexes5_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy5_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<final_flush_indexes5_t, boost::mpl::vector1<static_uint<3>>>::value), "Error");

    using iteration_policy6_t = iteration_policy<krange2::ToLevel, krange2::FromLevel, enumtype::backward>;

    using final_flush_indexes6_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy6_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<final_flush_indexes6_t, boost::mpl::vector1<static_uint<2>>>::value), "Error");

    using iteration_policy7_t = iteration_policy<krange1::ToLevel, krange1::FromLevel, enumtype::backward>;

    using final_flush_indexes7_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy7_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<final_flush_indexes7_t>::value == 0), "Error");

    using iteration_policy8_t = iteration_policy<kminimum::ToLevel, kminimum::FromLevel, enumtype::backward>;

    using final_flush_indexes8_t = iterate_domain_cache_t::kcache_final_flush_indexes<iteration_policy8_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal<final_flush_indexes8_t,
                                boost::mpl::vector3<static_uint<1>, static_uint<0>, static_uint<4>>>::value),
        "Error");
}

TEST(iterate_domain_cache, bpfill) {

    typedef detail::cache_impl<K, p_in1, cache_io_policy::fill, kminimum> cache1_t;
    typedef detail::cache_impl<K, p_in2, cache_io_policy::bpfill, kmin_and_range1> cache2_t;
    typedef detail::cache_impl<K, p_in3, cache_io_policy::bpfill, krange2_and_max> cache3_t;
    typedef detail::cache_impl<K, p_in4, cache_io_policy::local, kmaximum> cache4_t;
    typedef detail::cache_impl<K, p_out, cache_io_policy::flush, kall> cache5_t;

    typedef boost::mpl::vector5<cache1_t, cache2_t, cache3_t, cache4_t, cache5_t> caches_t;

    typedef std::tuple<p_in1, p_in2, p_in3, p_in4, p_out> esf_args_t;

    typedef local_domain<esf_args_t, extent<>, false> local_domain_t;

    typedef boost::mpl::vector2<extent<-1, 2, -2, 1>, extent<-2, 1, -3, 2>> extents_t;

    typedef typename boost::mpl::
        fold<extents_t, extent<0, 0, 0, 0>, enclosing_extent_2<boost::mpl::_1, boost::mpl::_2>>::type max_extent_t;

    typedef iterate_domain_arguments<backend_ids<platform::cuda, GRIDBACKEND, strategy::block>,
        local_domain_t,
        esfk_sequence_t,
        extents_t,
        max_extent_t,
        caches_t,
        gridtools::grid<axis_t>,
        boost::mpl::false_,
        notype>
        iterate_domain_arguments_t;

    using iterate_domain_cache_t = iterate_domain_cache<iterate_domain_arguments_t>;

    using k_filling_caches_indexes_t = iterate_domain_cache_t::k_filling_caches_indexes_t;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<k_filling_caches_indexes_t, boost::mpl::vector1<static_uint<0>>>::value), "Error");

    using iteration_policy1_t = iteration_policy<kminimum::FromLevel, kminimum::ToLevel, enumtype::forward>;

    using begin_fill_indexes1_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy1_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<begin_fill_indexes1_t, boost::mpl::vector2<static_uint<1>, static_uint<0>>>::value),
        "Error");

    using iteration_policy2_t = iteration_policy<krange1::FromLevel, krange1::ToLevel, enumtype::forward>;

    using begin_fill_indexes2_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy2_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<begin_fill_indexes2_t>::value == 0), "Error");

    using iteration_policy3_t = iteration_policy<krange2::FromLevel, krange2::ToLevel, enumtype::forward>;

    using begin_fill_indexes3_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy3_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<begin_fill_indexes3_t, boost::mpl::vector1<static_uint<2>>>::value), "Error");

    using iteration_policy4_t = iteration_policy<kmaximum::FromLevel, kmaximum::ToLevel, enumtype::forward>;

    using begin_fill_indexes4_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy4_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<begin_fill_indexes4_t>::value == 0), "Error");

    // backward
    using iteration_policy5_t = iteration_policy<kmaximum::ToLevel, kmaximum::FromLevel, enumtype::backward>;

    using begin_fill_indexes5_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy5_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<begin_fill_indexes5_t, boost::mpl::vector1<static_uint<2>>>::value), "Error");

    using iteration_policy6_t = iteration_policy<krange2::ToLevel, krange2::FromLevel, enumtype::backward>;

    using begin_fill_indexes6_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy6_t>::type;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<begin_fill_indexes6_t>::value == 0), "Error");

    using iteration_policy7_t = iteration_policy<krange1::ToLevel, krange1::FromLevel, enumtype::backward>;

    using begin_fill_indexes7_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy7_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<begin_fill_indexes7_t, boost::mpl::vector1<static_uint<1>>>::value), "Error");

    using iteration_policy8_t = iteration_policy<kminimum::ToLevel, kminimum::FromLevel, enumtype::backward>;

    using begin_fill_indexes8_t = iterate_domain_cache_t::kcache_begin_fill_indexes<iteration_policy8_t>::type;

    GRIDTOOLS_STATIC_ASSERT(
        (boost::mpl::equal<begin_fill_indexes8_t, boost::mpl::vector1<static_uint<0>>>::value), "Error");
}
