#pragma once
#include "iterate_domain.h"

namespace gridtools
{
    template<typename T> struct is_iterate_domain : boost::mpl::false_{};

    template<typename LocalDomain> struct is_iterate_domain<iterate_domain<LocalDomain> > : boost::mpl::true_{};

    template<typename LocalDomain>
    struct is_iterate_domain<stateful_iterate_domain<LocalDomain> > : boost::mpl::true_{};

    template<typename T>
    struct is_positional_iterate_domain : boost::mpl::false_{};

    template<typename LocalDomain>
    struct is_positional_iterate_domain<stateful_iterate_domain<LocalDomain> > : boost::mpl::true_{};

    template<typename T>
    struct iterate_domain_local_domain;

    template<typename LocalDomain>
    struct iterate_domain_local_domain<iterate_domain<LocalDomain> >
    {
        typedef LocalDomain type;
    };

    template<typename LocalDomain>
    struct iterate_domain_local_domain<stateful_iterate_domain<LocalDomain> >
    {
        typedef LocalDomain type;
    };

}
