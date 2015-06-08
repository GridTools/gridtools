#pragma once

#include "../iterate_domain.h"
#include "../iterate_domain_metafunctions.h"

namespace gridtools {

/**
 * @brief iterate domain class for the Host backend
 */
    template<typename LocalDomain, bool IsPositional>
    class iterate_domain_host : public iterate_domain<iterate_domain_host<LocalDomain, IsPositional>, IsPositional > //CRTP
{
    DISALLOW_COPY_AND_ASSIGN(iterate_domain_host);
    typedef iterate_domain<iterate_domain_host<LocalDomain, IsPositional>, IsPositional > super;
public:
    typedef LocalDomain local_domain_t;

    GT_FUNCTION
    explicit iterate_domain_host(LocalDomain const& local_domain)
        : super(local_domain) {}

};

template<typename LocalDomain, bool IsPositional>
struct is_iterate_domain<iterate_domain_host<LocalDomain, IsPositional> > : public boost::mpl::true_{};

}
