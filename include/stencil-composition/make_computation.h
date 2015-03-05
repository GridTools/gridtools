#pragma once

#include <boost/ref.hpp>
#ifndef __CUDACC__
#include <boost/make_shared.hpp>
#endif
#include "intermediate.h"
#include "../common/meta_array.h"

namespace gridtools {
#ifdef __CUDACC__
    template <typename Backend, typename LayoutType, typename MssType, typename Domain, typename Coords>
    computation* make_computation(typename mss_array<boost::mpl::vector1<MssType> >::type const& mss, Domain & domain, Coords const& coords) {
        return new intermediate<Backend, LayoutType, typename mss_array<boost::mpl::vector1<MssType> >::type, Domain, Coords>(mss, boost::ref(domain), coords);
    }
#else
    template <typename Backend, typename LayoutType, typename MssType, typename Domain, typename Coords>
    boost::shared_ptr<intermediate<Backend, LayoutType, typename mss_array<boost::mpl::vector1<MssType> >::type, Domain, Coords>/*computation*/> make_computation(MssType const& mss, Domain & domain, Coords const& coords) {
        return boost::make_shared<intermediate<Backend, LayoutType, typename mss_array<boost::mpl::vector1<MssType> >::type, Domain, Coords> >(boost::ref(domain), coords);
    }

    template <typename Backend, typename LayoutType, typename MssType1, typename MssType2, typename Domain, typename Coords>
    boost::shared_ptr<intermediate<Backend, LayoutType, typename mss_array<boost::mpl::vector2<MssType1, MssType2> >::type, Domain, Coords>/*computation*/> make_computation(MssType1 const& mss1, MssType2 const& mss2, Domain & domain, Coords const& coords) {
        return boost::make_shared<intermediate<Backend, LayoutType, typename mss_array<boost::mpl::vector2<MssType1, MssType2> >::type, Domain, Coords> >(boost::ref(domain), coords);
    }

#endif
} //namespace gridtools
