#pragma once

#include <boost/ref.hpp>
#ifndef __CUDACC__
#include <boost/make_shared.hpp>
#endif
#include "intermediate.h"

namespace gridtools {
#ifdef __CUDACC__
    template <typename Backend, typename LayoutType, typename MssType, typename Domain, typename Coords>
    computation* make_computation(MssType const& mss, Domain & domain, Coords const& coords) {
        return new intermediate<Backend, LayoutType, MssType, Domain, Coords>(mss, boost::ref(domain), coords);
    }
#else
    template <typename Backend, typename LayoutType, typename MssType, typename Domain, typename Coords>
    boost::shared_ptr<intermediate<Backend, LayoutType, MssType, Domain, Coords>/*computation*/> make_computation(MssType const& mss, Domain & domain, Coords const& coords) {
        return boost::make_shared<intermediate<Backend, LayoutType, MssType, Domain, Coords> >(mss, boost::ref(domain), coords);
    }
#endif
} //namespace gridtools
