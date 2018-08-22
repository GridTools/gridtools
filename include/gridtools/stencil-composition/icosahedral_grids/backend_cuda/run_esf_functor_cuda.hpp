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
#pragma once

#include <type_traits>

#include <boost/mpl/at.hpp>

#include "../../../common/defs.hpp"
#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../common/generic_metafunctions/meta.hpp"
#include "../../../common/generic_metafunctions/type_traits.hpp"
#include "../../../common/host_device.hpp"
#include "../../functor_decorator.hpp"
#include "../../location_type.hpp"
#include "../../run_esf_functor.hpp"
#include "../../run_functor_arguments.hpp"
#include "../esf.hpp"
#include "../iterate_domain_remapper.hpp"

namespace gridtools {

    /*
     * @brief main functor that executes (for CUDA) the user functor of an ESF
     * @tparam RunFunctorArguments run functor arguments
     * @tparam Interval interval where the functor gets executed
     */
    struct run_esf_functor_cuda {
      private:
        template <typename Esf>
        struct esf_has_color : bool_constant<!std::is_same<typename Esf::color_t, nocolor>::value> {
            GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor<Esf>::value), GT_INTERNAL_ERROR);
        };

        template <class IntervalType, class EsfArguments, uint_t Color, class ItDomain>
        GT_FUNCTION_DEVICE static void call_colored(ItDomain &it_domain) {
            using esf_t = typename EsfArguments::esf_t;
            using iterate_domain_remapper_t = typename get_iterate_domain_remapper<ItDomain,
                typename EsfArguments::esf_t::args_t,
                typename esf_get_location_type<esf_t>::type,
                Color>::type;

            iterate_domain_remapper_t iterate_domain_remapper(it_domain);

            using original_functor_t = typename EsfArguments::functor_t;
            using functor_decorator_t = functor_decorator<typename original_functor_t::id,
                typename esf_t::template esf_function<Color>,
                typename original_functor_t::repeat_t,
                IntervalType>;
            // call the user functor at the core of the block
            call_repeated<functor_decorator_t, IntervalType>(iterate_domain_remapper);
        }

        template <class IntervalType, class EsfArguments, class ItDomain>
        struct color_functor {
            ItDomain &m_it_domain;
            template <typename Color>
            GT_FUNCTION_DEVICE void operator()(Color) const {
                call_colored<IntervalType, EsfArguments, Color::value>(m_it_domain);
                m_it_domain.increment_c();
            }
        };

      public:
        /*
         * @brief main functor implemenation that executes (for CUDA) the user functor of an ESF
         * @tparam IntervalType interval where the functor gets executed
         * @tparam EsfArgument esf arguments type that contains the arguments needed to execute this ESF.
         */

        // specialization of the loop over colors when the user specified the ESF with a specific color
        // Only that color gets executed
        template <class IntervalType,
            class EsfArguments,
            class ItDomain,
            class Esf = typename EsfArguments::esf_t,
            enable_if_t<esf_has_color<Esf>::value, int> = 0>
        GT_FUNCTION_DEVICE void operator()(ItDomain &it_domain) const {
            GRIDTOOLS_STATIC_ASSERT(is_esf_arguments<EsfArguments>::value, GT_INTERNAL_ERROR);
            // a grid point at the core of the block can be out of extent (for last blocks) if domain of computations
            // is not a multiple of the block size
            if (it_domain.template is_thread_in_domain<typename EsfArguments::extent_t>()) {
                static constexpr auto color = Esf::color_t::value;
                // TODO we could identify if previous ESF was in the same color and avoid this iterator operations
                it_domain.template increment_c<color>();
                call_colored<IntervalType, EsfArguments, color>(it_domain);
                it_domain.template increment_c<-color>();
            }
            // synchronize threads if not independent esf
            if (!boost::mpl::at<typename EsfArguments::async_esf_map_t, typename EsfArguments::functor_t>::type::value)
                __syncthreads();
        }

        // specialization of the loop over colors when the ESF does not specify any particular color.
        // A loop over all colors is performed.
        template <class IntervalType,
            class EsfArguments,
            class ItDomain,
            class Esf = typename EsfArguments::esf_t,
            enable_if_t<!esf_has_color<Esf>::value, int> = 0>
        GT_FUNCTION_DEVICE void operator()(ItDomain &it_domain) const {
            GRIDTOOLS_STATIC_ASSERT(is_esf_arguments<EsfArguments>::value, GT_INTERNAL_ERROR);
            // a grid point at the core of the block can be out of extent (for last blocks) if domain of computations
            // is not a multiple of the block size
            if (it_domain.template is_thread_in_domain<typename EsfArguments::extent_t>()) {
                // loop over colors executing user functor for each color
                static constexpr auto NColors = esf_get_location_type<Esf>::type::n_colors::value;
                gridtools::for_each<GT_META_CALL(meta::make_indices_c, NColors)>(
                    color_functor<IntervalType, EsfArguments, ItDomain>{it_domain});
                it_domain.template increment_c<-NColors>();
            }
            // synchronize threads if not independent esf
            if (!boost::mpl::at<typename EsfArguments::async_esf_map_t, typename EsfArguments::functor_t>::type::value)
                __syncthreads();
        }
    };
} // namespace gridtools
