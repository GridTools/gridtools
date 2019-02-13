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

#include <tuple>

#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/reverse_fold.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/vector.hpp>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/is_sequence_of.hpp"
#include "linearize_mss_functions.hpp"
#include "mss.hpp"
#include "mss_components.hpp"

namespace gridtools {

    template <typename MssDescriptor>
    struct mss_split_esfs {
        GT_STATIC_ASSERT(is_mss_descriptor<MssDescriptor>::value, GT_INTERNAL_ERROR);

        using execution_engine_t = typename mss_descriptor_execution_engine<MssDescriptor>::type;

        template <typename Esf_>
        struct compose_mss_ {
            using type = mss_descriptor<execution_engine_t, std::tuple<Esf_>>;
        };

        using mss_split_multiple_esf_t =
            typename boost::mpl::fold<typename mss_descriptor_linear_esf_sequence<MssDescriptor>::type,
                boost::mpl::vector0<>,
                boost::mpl::push_back<boost::mpl::_1, compose_mss_<boost::mpl::_2>>>::type;

        using type = typename boost::mpl::if_c<
            // if the number of esf contained in the mss is 1, there is no need to split
            boost::mpl::size<typename mss_descriptor_linear_esf_sequence<MssDescriptor>::type>::value == 1,
            boost::mpl::vector1<MssDescriptor>,
            mss_split_multiple_esf_t>::type;
    };

    // TODOCOSUNA unittest this
    /**
     * @brief metafunction that takes an MSS with multiple ESFs and split it into multiple MSS with one ESF each
     * Only to be used for CPU. GPU always fuses ESFs and there is no clear way to split the caches.
     * @tparam Msses computaion token sequence
     */
    template <typename Msses>
    struct split_mss_into_independent_esfs {
        GT_STATIC_ASSERT((is_sequence_of<Msses, is_mss_descriptor>::value), GT_INTERNAL_ERROR);

        typedef typename boost::mpl::reverse_fold<Msses,
            boost::mpl::vector0<>,
            boost::mpl::copy<boost::mpl::_1, boost::mpl::back_inserter<mss_split_esfs<boost::mpl::_2>>>>::type type;
    };

    /**
     * @brief metafunction that builds the array of mss components
     * @tparam BackendId id of the backend (which decides whether the MSS with multiple ESF are split or not)
     * @tparam MssDescriptors mss descriptor sequence
     * @tparam extent_sizes sequence of sequence of extents
     */
    template <typename MssFuseEsfStrategy, typename MssDescriptors, typename ExtentMap, typename Axis>
    struct build_mss_components_array {
        GT_STATIC_ASSERT((is_sequence_of<MssDescriptors, is_mss_descriptor>::value), GT_INTERNAL_ERROR);

        using mss_seq_t = typename boost::mpl::eval_if<MssFuseEsfStrategy,
            boost::mpl::identity<MssDescriptors>,
            split_mss_into_independent_esfs<MssDescriptors>>::type;

        using type = typename boost::mpl::transform<mss_seq_t, mss_components<boost::mpl::_, ExtentMap, Axis>>::type;
    };

} // namespace gridtools
