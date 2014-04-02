#pragma once

#include <boost/ref.hpp>
#include <boost/make_shared.hpp>
#include "intermediate.h"

namespace gridtools {
    template <typename Backend, typename MssType, typename Domain, typename Coords>
    boost::shared_ptr<computation> make_computation(MssType const& mss, Domain & domain, Coords const& coords) {
        return boost::make_shared<intermediate<Backend, MssType, Domain, Coords> >(mss, boost::ref(domain), coords);
    }

} //namespace gridtools
