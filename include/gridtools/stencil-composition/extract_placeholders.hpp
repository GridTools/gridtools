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

#include "../common/defs.hpp"

#include "../common/generic_metafunctions/copy_into_variadic.hpp"

#include "independent_esf.hpp"

#if GT_BROKEN_TEMPLATE_ALIASES

#include <boost/mpl/fold.hpp>
#include <boost/mpl/empty_sequence.hpp>
#include <boost/mpl/quote.hpp>
#include <boost/mpl/joint_view.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/single_view.hpp>

namespace gridtools {
    namespace _impl {

        // Flatten an MPL sequence of MPL sequences.
        template < class Seqs >
        struct flatten
            : boost::mpl::fold< Seqs, boost::mpl::empty_sequence, boost::mpl::quote2< boost::mpl::joint_view > > {};

        // Deduplicate an MPL sequence.
        template < class Seq >
        struct dedup
            : boost::mpl::fold< Seq, boost::mpl::set0<>, boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > > {};

        template < class SeqOfTrees, template < class > class FlattenTree >
        struct flatten_trees : flatten< boost::mpl::transform_view< SeqOfTrees, boost::mpl::quote1< FlattenTree > > > {
        };

        //  Flatten an ESF tree formed by regular ESF's and independent_esf into an MPL sequence.
        //  The result contains only regular ESF's.
        template < class T >
        struct flatten_esf {
            using type = boost::mpl::single_view< T >;
        };
        template < class T >
        struct flatten_esfs : flatten_trees< T, flatten_esf > {};
        template < class EsfSeq >
        struct flatten_esf< independent_esf< EsfSeq > > {
            using type = typename flatten_esfs< EsfSeq >::type;
        };

        // Extract ESFs from an MSS.
        template < class Mss >
        struct get_esfs : flatten_esfs< typename Mss::esf_sequence_t > {};

        // Extract args from ESF.
        template < class Esf >
        struct get_args {
            using type = typename Esf::args_t;
        };
    }

    /// Takes an MPL sequence of MSS descriptions and returns deduplicated typelist of all placeholders
    /// that are used in the given msses.
    template < class Msses >
    using extract_placeholders = copy_into_variadic< typename _impl::dedup< typename _impl::flatten_trees<
                                                         typename _impl::flatten_trees< Msses, _impl::get_esfs >::type,
                                                         _impl::get_args >::type >::type,
        std::tuple<> >;
}

#else

#include "../common/generic_metafunctions/meta.hpp"

namespace gridtools {
    namespace _impl {
        template < class Trees, template < class... > class FlattenTree >
        using flatten_trees = meta::flatten< meta::transform< FlattenTree, Trees > >;

        //  Flatten an ESF tree formed by regular ESF's and independent_esf into an MPL sequence.
        //  The result contains only regular ESF's.
        template < class >
        struct lazy_flatten_esf;
        template < class T >
        using flatten_esf = typename lazy_flatten_esf< T >::type;
        template < class T >
        using flatten_esfs = flatten_trees< T, flatten_esf >;
        template < class T >
        struct lazy_flatten_esf {
            using type = std::tuple< T >;
        };
        template < class EsfSeq >
        struct lazy_flatten_esf< independent_esf< EsfSeq > > {
            using type = flatten_esfs< EsfSeq >;
        };

        // Extract ESFs from an MSS.
        template < class Mss >
        using get_esfs = flatten_esfs< copy_into_variadic< typename Mss::esf_sequence_t, std::tuple<> > >;

        // Extract args from ESF.
        template < class Esf >
        using get_args = copy_into_variadic< typename Esf::args_t, std::tuple<> >;
    }

    /// Takes a typelist of MSS descriptions and returns deduplicated typelist of all placeholders
    /// that are used in the given msses.
    template < class Msses >
    using extract_placeholders =
        meta::dedup< _impl::flatten_trees< _impl::flatten_trees< Msses, _impl::get_esfs >, _impl::get_args > >;
}

#endif
