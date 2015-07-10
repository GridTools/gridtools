#pragma once

// Copyright Aleksey Gurtovoy 2000-2008
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/mpl for documentation.

// $Id: for_each.hpp 55648 2009-08-18 05:16:53Z agurtovoy $
// $Date: 2009-08-18 01:16:53 -0400 (Tue, 18 Aug 2009) $
// $Revision: 55648 $

#include <boost/version.hpp>

#if BOOST_VERSION >= 105600
#include <boost/mpl/for_each.hpp>
namespace gridtools {
    using boost::mpl::for_each;
}
#else
#include "../common/host_device.hpp"
#define BOOST_MPL_GPU_ENABLED __host__ __device__

#include <boost/static_assert.hpp>
#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/apply.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/next_prior.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/assert.hpp>
#include "unwrap.hpp"

#include <boost/type_traits/is_same.hpp>
#include "value_init.hpp"

namespace gridtools {

    namespace gt_aux {

        template< bool Done = true >
        struct for_each_impl
        {
            template<
                typename Iterator
                , typename LastIterator
                , typename TransformFunc
                , typename F
                >
            BOOST_MPL_GPU_ENABLED
            static void execute(
                                Iterator*
                                , LastIterator*
                                , TransformFunc*
                                , F
                                )
            {
            }
        };

        template<>
        struct for_each_impl<false>
        {
            template<
                typename Iterator
                , typename LastIterator
                , typename TransformFunc
                , typename F
                >
            BOOST_MPL_GPU_ENABLED
            static void execute(
                                Iterator*
                                , LastIterator*
                                , TransformFunc*
                                , F f
                                )
            {
                typedef typename boost::mpl::deref<Iterator>::type item;
                typedef typename boost::mpl::apply1<TransformFunc,item>::type arg;

                // dwa 2002/9/10 -- make sure not to invoke undefined behavior
                // when we pass arg.
                value_initialized<arg> x;
                gt_aux::unwrap(f, 0)(get(x));

                typedef typename boost::mpl::next<Iterator>::type iter;
                for_each_impl<boost::is_same<iter,LastIterator>::value>
                    ::execute( static_cast<iter*>(0), static_cast<LastIterator*>(0), static_cast<TransformFunc*>(0), f);
            }
        };

    } // namespace gt_aux

    // agurt, 17/mar/02: pointer default parameters are necessary to workaround
    // MSVC 6.5 function template signature's mangling bug
    template<
        typename Sequence
        , typename F
        >
    BOOST_MPL_GPU_ENABLED
    inline
    void for_each(F f, Sequence* = 0)
    {
        BOOST_STATIC_ASSERT( boost::mpl::is_sequence<Sequence>::value );

        typedef typename boost::mpl::begin<Sequence>::type first;
        typedef typename boost::mpl::end<Sequence>::type last;

        gt_aux::for_each_impl< boost::is_same<first,last>::value >
            ::execute(static_cast<first*>(0), static_cast<last*>(0), static_cast<boost::mpl::identity<>*>(0), f);
    }

}
#endif
