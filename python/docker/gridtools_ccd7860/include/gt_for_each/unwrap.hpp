#pragma once

// Copyright Peter Dimov and Multi Media Ltd 2001, 2002
// Copyright David Abrahams 2001
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/mpl for documentation.

// $Id: unwrap.hpp 49267 2008-10-11 06:19:02Z agurtovoy $
// $Date: 2008-10-11 02:19:02 -0400 (Sat, 11 Oct 2008) $
// $Revision: 49267 $

#include <boost/ref.hpp>
#include <common/defs.hpp>

namespace gridtools {
    namespace gt_aux {

        template< typename F >
        BOOST_MPL_GPU_ENABLED
        inline
        F& unwrap(F& f, long int_t)
        {
            return f;
        }

        template< typename F >
        BOOST_MPL_GPU_ENABLED
        inline
        F&
        unwrap(boost::reference_wrapper<F>& f, int_t)
        {
            return f;
        }

        template< typename F >
        BOOST_MPL_GPU_ENABLED
        inline
        F&
        unwrap(boost::reference_wrapper<F> const& f, int_t)
        {
            return f;
        }

    } // namespace gt_aux
} // namespace gridtools
