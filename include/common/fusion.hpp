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
/// An unstructured set of boost::fusion related helpers
#pragma once
#include <tuple>
#include <type_traits>
#include <utility>

#include <boost/mpl/identity.hpp>
#include <boost/mpl/transform_view.hpp>
#include <boost/fusion/include/category_of.hpp>
#include <boost/fusion/include/move.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/filter_view.hpp>
#include <boost/fusion/include/joint_view.hpp>
#include <boost/fusion/include/std_tuple.hpp>
#include <boost/fusion/include/transform_view.hpp>
#include <boost/fusion/include/zip_view.hpp>

#include "generic_metafunctions/copy_into_variadic.hpp"
#include "generic_metafunctions/gt_integer_sequence.hpp"

namespace gridtools {

    /**
     *  Here go generators for the fusion views that do the right C++11 perfect forwarding.
     */

    template < typename Lhs, typename Rhs >
    boost::fusion::joint_view< typename std::remove_reference< Lhs >::type,
        typename std::remove_reference< Rhs >::type >
    make_joint_view(Lhs &&lhs, Rhs &&rhs) {
        return {lhs, rhs};
    }

    template < typename Pred, typename Seq >
    boost::fusion::filter_view< typename std::remove_reference< Seq >::type, Pred > make_filter_view(Seq &&seq) {
        return {std::forward< Seq >(seq)};
    };

    template < typename Seqs >
    boost::fusion::zip_view< typename std::remove_reference< Seqs >::type > make_zip_view(Seqs &&secs) {
        return {std::forward< Seqs >(secs)};
    }

    template < typename Seq, typename F >
    boost::fusion::transform_view< typename std::remove_reference< Seq >::type, F > make_transform_view(
        Seq &&seq, F const &f) {
        return {std::forward< Seq >(seq), f};
    }

    namespace _impl {
        template < class F >
        struct generator_f {
            F const &m_f;
            template < class T >
            T operator()(boost::mpl::identity< T >) const {
                return m_f.template operator()< T >();
            }
#ifndef BOOST_RESULT_OF_USE_DECLTYPE
            template < class >
            class result;

            template < class T >
            class result< generator_f(boost::mpl::identity< T > const &) > {
                using type = T;
            };
#endif
        };
    }

    template < typename Seq, typename F >
    Seq generate_sequence(F const &f) {
        using identities = typename boost::fusion::result_of::as_vector<
            boost::mpl::transform_view< Seq, boost::mpl::make_identity< boost::mpl::_ > > >::type;
        return boost::fusion::transform_view< const identities, _impl::generator_f< F > >({}, {f});
    }

    template < class... Ts >
    std::tuple< Ts... > &as_std_tuple(std::tuple< Ts... > &seq) {
        return seq;
    }

    template < class... Ts >
    std::tuple< Ts... > const &as_std_tuple(std::tuple< Ts... > const &seq) {
        return seq;
    }

    template < class... Ts >
    std::tuple< Ts... > as_std_tuple(std::tuple< Ts... > &&seq) {
        return seq;
    }

    template < typename Seq,
        class Decayed = typename std::decay< Seq >::type,
        typename Res = copy_into_variadic< Decayed, std::tuple<> > >
    Res as_std_tuple(Seq &&seq) {
        Res res;
        boost::fusion::move(std::forward< Seq >(seq), res);
        return res;
    }
}
