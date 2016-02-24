#pragma once
#include <boost/make_shared.hpp>
#include "../stencil-composition/computation.hpp"
#include "../stencil-composition/intermediate.hpp"

namespace gridtools {
    template <
        typename Backend,
        typename Esf,
        typename Domain,
        typename Grid
    >
    boost::shared_ptr<computation> make_reduction(
        Esf esf,
        Domain& domain, const Grid& grid
    ) {
        return boost::make_shared<
            intermediate<
                Backend,
                meta_array<
                    boost::mpl::vector1< mss_descriptor<enumtype::execute<enumtype::forward>, boost::mpl::vector1<Esf> > >,
                    boost::mpl::quote1<is_mss_descriptor>
                >,
                Domain, Grid, false
            >
         >(boost::ref(domain), grid);
    }

} // namespace gridtools
