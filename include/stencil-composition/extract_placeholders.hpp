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

#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/empty_sequence.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/joint_view.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/single_view.hpp>
#include <boost/mpl/transform_view.hpp>
#include <boost/mpl/vector.hpp>

namespace gridtools {
    namespace _impl {
        // Flatten an MPL sequence of MPL sequences.
        template < class Seqs >
        using flatten_t = typename boost::mpl::fold< Seqs,
            boost::mpl::empty_sequence,
            boost::mpl::quote2< boost::mpl::joint_view > >::type;

        // Deduplicate an MPL sequence.
        template < class Seq >
        using dedup_t = typename boost::mpl::fold< Seq,
            boost::mpl::set0<>,
            boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > >::type;

        template < class SeqOfTrees, template < class > class FlattenTree >
        using flatten_trees_t =
            flatten_t< boost::mpl::transform_view< SeqOfTrees, boost::mpl::quote1< FlattenTree > > >;

        //  Flatten an ESF tree formed by regular ESF's and independent_esf into an MPL sequence.
        //  The result contains only regular ESF's.
        template < class >
        struct flatten_esf;
        template < class T >
        using flatten_esf_t = typename flatten_esf< T >::type;
        template < class T >
        using flatten_esfs_t = flatten_trees_t< T, flatten_esf >;
        template < class T >
        struct flatten_esf {
            using type = boost::mpl::single_view< T >;
        };
        template < class EsfSeq >
        struct flatten_esf< independent_esf< EsfSeq > > {
            using type = flatten_esfs_t< EsfSeq >;
        };

        // Extract ESFs from an MSS.
        template < class Mss >
        struct esfs {
            using type = flatten_esfs_t< typename Mss::esf_sequence_t >;
        };

        // Extract args from ESF.
        template < class Esf >
        struct args {
            using type = typename Esf::args_t;
        };

        template < class Msses >
        struct extract_placeholders {
            using esfs_t = flatten_trees_t< Msses, esfs >;
            using raw_args_t = dedup_t< flatten_trees_t< esfs_t, args > >;
            using type =
                typename boost::mpl::copy< raw_args_t, boost::mpl::back_inserter< boost::mpl::vector0<> > >::type;
        };
    }

    template < class Msses >
    using extract_placeholders = _impl::extract_placeholders< Msses >;
}
