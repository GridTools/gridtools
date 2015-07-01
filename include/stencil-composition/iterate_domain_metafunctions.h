#pragma once
#include "iterate_domain.h"

namespace gridtools
{
    template<typename T> struct is_iterate_domain : boost::mpl::false_{};

    template<typename T>
    struct is_positional_iterate_domain : boost::mpl::false_{};

    template<typename IterateDomainImpl>
    struct is_positional_iterate_domain<positional_iterate_domain<IterateDomainImpl> > : boost::mpl::true_{};

    template<typename T>
    struct iterate_domain_local_domain;

    template<
        template<template<class> class, typename> class IterateDomainImpl,
        template<class> class IterateDomainBase,
        typename LocalDomain
    >
    struct iterate_domain_local_domain<IterateDomainImpl<IterateDomainBase, LocalDomain> >
    {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain<IterateDomainImpl<IterateDomainBase, LocalDomain> >::value),
                "Internal Error: wrong type")
        typedef LocalDomain type;
    };
}
