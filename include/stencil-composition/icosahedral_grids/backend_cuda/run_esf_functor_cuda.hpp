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
#include <boost/utility/enable_if.hpp>
#include "../../run_esf_functor.hpp"
#include "../../block_size.hpp"
#include "../iterate_domain_remapper.hpp"
#include "../../functor_decorator.hpp"

namespace gridtools {

    template < typename Esf >
    struct esf_has_color {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< Esf >::value), GT_INTERNAL_ERROR);
        typedef typename boost::mpl::not_< typename boost::is_same< typename Esf::color_t, nocolor >::type >::type type;
        static const bool value = type::value;
    };

    template < typename Esf >
    struct esf_color_range {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< Esf >::value), GT_INTERNAL_ERROR);
        template < typename Esf_ >
        struct build_range_ {
            typedef boost::mpl::range_c< uint_t, Esf_::color_t::color_t::value, Esf_::color_t::color_t::value > type;
        };
        template < typename Esf_ >
        struct build_full_range_ {
            typedef boost::mpl::range_c< uint_t, 0, esf_get_location_type< Esf_ >::type::n_colors::value > type;
        };

        typedef
            typename boost::mpl::eval_if< esf_has_color< Esf >, build_range_< Esf >, build_full_range_< Esf > >::type
                type;
    };

    template < typename IterateDomain, typename EsfArguments, typename EsfLocationType, typename IntervalType >
    struct color_functor {
        GRIDTOOLS_STATIC_ASSERT((is_location_type< EsfLocationType >::value), GT_INTERNAL_ERROR);

      private:
        IterateDomain &m_iterate_domain;

      public:
        GT_FUNCTION
        color_functor(IterateDomain &iterate_domain) : m_iterate_domain(iterate_domain) {}

        template < typename Index >
        GT_FUNCTION void operator()(Index const &) {

            typedef typename get_iterate_domain_remapper< IterateDomain,
                typename EsfArguments::esf_args_map_t,
                EsfLocationType,
                Index::value >::type iterate_domain_remapper_t;

            typedef typename EsfArguments::esf_t esf_t;

            iterate_domain_remapper_t iterate_domain_remapper(m_iterate_domain);

            // typedef typename esf_t::template esf_function< Index::value > functor_t;

            typedef typename EsfArguments::functor_t original_functor_t;
            typedef typename esf_t::template esf_function< Index::value > colored_functor_t;
            typedef functor_decorator< typename original_functor_t::id,
                colored_functor_t,
                typename original_functor_t::repeat_t,
                IntervalType > functor_t;

            GRIDTOOLS_STATIC_ASSERT(is_functor_decorator< functor_t >::value, GT_INTERNAL_ERROR);

            // call the user functor at the core of the block
            _impl::call_repeated< functor_t::repeat_t::value, functor_t, iterate_domain_remapper_t, IntervalType >::
                call_do_method(iterate_domain_remapper);
            (m_iterate_domain)
                .template increment< grid_traits_from_id< enumtype::icosahedral >::dim_c_t::value, static_uint< 1 > >();
        }
    };

    /*
     * @brief main functor that executes (for CUDA) the user functor of an ESF
     * @tparam RunFunctorArguments run functor arguments
     * @tparam Interval interval where the functor gets executed
     */
    template < typename RunFunctorArguments, typename Interval >
    struct run_esf_functor_cuda
        : public run_esf_functor< run_esf_functor_cuda< RunFunctorArguments, Interval > > // CRTP
    {
        GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArguments >::value), GT_INTERNAL_ERROR);
        // TODOCOSUNA This type here is not an interval, is a pair<int_, int_ >
        // BOOST_STATIC_ASSERT((is_interval<Interval>::value));

        typedef run_esf_functor< run_esf_functor_cuda< RunFunctorArguments, Interval > > super;
        typedef typename RunFunctorArguments::physical_domain_block_size_t physical_domain_block_size_t;
        typedef typename RunFunctorArguments::processing_elements_block_size_t processing_elements_block_size_t;

        // metavalue that determines if a warp is processing more grid points that the default assigned
        // at the core of the block
        typedef typename boost::mpl::not_< typename boost::is_same< physical_domain_block_size_t,
            processing_elements_block_size_t >::type >::type multiple_grid_points_per_warp_t;

        // nevertheless, even if each thread computes more than a grid point, the i size of the physical block
        // size and the cuda block size have to be the same
        GRIDTOOLS_STATIC_ASSERT(
            (physical_domain_block_size_t::i_size_t::value == processing_elements_block_size_t::i_size_t::value),
            GT_INTERNAL_ERROR);

        typedef typename RunFunctorArguments::iterate_domain_t iterate_domain_t;

        using super::m_iterate_domain;

        GT_FUNCTION
        explicit run_esf_functor_cuda(iterate_domain_t &iterate_domain) : super(iterate_domain) {}

        /*
         * @brief main functor implemenation that executes (for CUDA) the user functor of an ESF
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         */
        template < typename IntervalType, typename EsfArguments >
        __device__ void do_impl() const {
            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments< EsfArguments >::value), GT_INTERNAL_ERROR);

            typedef typename EsfArguments::functor_t functor_t;
            typedef typename EsfArguments::extent_t extent_t;

            // a grid point at the core of the block can be out of extent (for last blocks) if domain of computations
            // is not a multiple of the block size
            if (m_iterate_domain.template is_thread_in_domain< extent_t >()) {
                // loop over colors excuting user funtor for each color
                color_loop< IntervalType, EsfArguments >();
            }

            // synchronize threads if not independent esf
            if (!boost::mpl::at< typename EsfArguments::async_esf_map_t, functor_t >::type::value)
                __syncthreads();
        }

      private:
        // specialization of the loop over colors when the user speficied the ESF with a specific color
        // Only that color gets executed
        template < typename IntervalType, typename EsfArguments >
        __device__ void color_loop(
            typename boost::enable_if< typename esf_has_color< typename EsfArguments::esf_t >::type, int >::type =
                0) const {

            typedef typename EsfArguments::esf_t::color_t::color_t color_t;
            typedef typename esf_get_location_type< typename EsfArguments::esf_t >::type location_type_t;

            typedef typename get_iterate_domain_remapper< iterate_domain_t,
                typename EsfArguments::esf_args_map_t,
                location_type_t,
                color_t::value >::type iterate_domain_remapper_t;

            iterate_domain_remapper_t iterate_domain_remapper(m_iterate_domain);

            typedef typename EsfArguments::esf_t esf_t;
            typedef typename esf_t::template esf_function< color_t::value > functor_t;

            GRIDTOOLS_STATIC_ASSERT(is_functor_decorator< functor_t >::value, GT_INTERNAL_ERROR);

            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments< EsfArguments >::value), GT_INTERNAL_ERROR);

            // TODO we could identify if previous ESF was in the same color and avoid this iterator operations
            (m_iterate_domain)
                .template increment< grid_traits_from_id< enumtype::icosahedral >::dim_c_t::value, color_t >();

            // call the user functor at the core of the block
            _impl::call_repeated< functor_t::repeat_t::value, functor_t, iterate_domain_remapper_t, IntervalType >::
                call_do_method(iterate_domain_remapper);
            (m_iterate_domain)
                .template increment< grid_traits_from_id< enumtype::icosahedral >::dim_c_t::value,
                    static_int< -color_t::value > >();
        }

        // specialization of the loop over colors when the ESF does not specify any particular color.
        // A loop over all colors is performed.
        template < typename IntervalType, typename EsfArguments >
        __device__ void color_loop(
            typename boost::disable_if< typename esf_has_color< typename EsfArguments::esf_t >::type, int >::type =
                0) const {

            typedef typename esf_get_location_type< typename EsfArguments::esf_t >::type location_type_t;

            GRIDTOOLS_STATIC_ASSERT((is_esf_arguments< EsfArguments >::value), GT_INTERNAL_ERROR);

            typedef typename esf_color_range< typename EsfArguments::esf_t >::type color_range_t;

            boost::mpl::for_each< color_range_t >(
                color_functor< iterate_domain_t, EsfArguments, location_type_t, IntervalType >(m_iterate_domain));

            using neg_n_colors_t = static_uint< -location_type_t::n_colors::value >;
            (m_iterate_domain)
                .template increment< grid_traits_from_id< enumtype::icosahedral >::dim_c_t::value, neg_n_colors_t >();
        }
    };
}
