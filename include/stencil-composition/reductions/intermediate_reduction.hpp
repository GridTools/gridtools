#pragma once
#include "../stencil-composition/intermediate.hpp"

namespace gridtools {

    /**
     * @class
     *  @brief structure collecting helper metafunctions
     */
    template < typename Backend, typename MssDescriptorArray, typename DomainType, typename Grid >
    struct intermediate_reduction : public intermediate<Backend, MssDescriptorArray, DomainType, Grid, false> {

        typedef intermediate<Backend, MssDescriptorArray, DomainType, Grid, false> base_t;
        typedef typename base_t::mss_components_array_t mss_components_array_t;
        typedef typename base_t::mss_local_domains_t mss_local_domains_t;
        using base_t::m_meter;
        using base_t::m_grid;
        using base_t::m_mss_local_domain_list;

        explicit intermediate_reduction(DomainType &domain, Grid const &grid) : base_t(domain, grid) {}

        virtual void run() {}
        /**
        * \brief the execution of the stencil operations take place in this call
        *
        */
        virtual void reduce() {

            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< typename mss_components_array_t::elements >::value ==
                                        boost::mpl::size< mss_local_domains_t >::value),
                "Internal Error");
            m_meter.start();
            Backend::template run< mss_components_array_t >(m_grid, m_mss_local_domain_list);
            m_meter.pause();
        }
    };
}
