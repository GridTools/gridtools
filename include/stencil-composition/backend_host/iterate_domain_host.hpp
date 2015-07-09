#pragma once

#include "../iterate_domain.hpp"
#include "../iterate_domain_metafunctions.hpp"

namespace gridtools {

/**
 * @brief iterate domain class for the Host backend
 */
template<template<class> class IterateDomainBase, typename LocalDomain>
class iterate_domain_host : public IterateDomainBase<iterate_domain_host<IterateDomainBase, LocalDomain> > //CRTP
{
    DISALLOW_COPY_AND_ASSIGN(iterate_domain_host);
    typedef IterateDomainBase<iterate_domain_host<IterateDomainBase, LocalDomain> > super;
public:
    GT_FUNCTION
    explicit iterate_domain_host(LocalDomain const& local_domain)
        : super(local_domain) {}

};

template<
    template<class> class IterateDomainBase, typename LocalDomain>
struct is_iterate_domain<
    iterate_domain_host<IterateDomainBase, LocalDomain>
> : public boost::mpl::true_{};

template<
    template<class> class IterateDomainBase,
    typename LocalDomain
>
struct is_positional_iterate_domain<iterate_domain_host<IterateDomainBase, LocalDomain> > :
    is_positional_iterate_domain<IterateDomainBase<iterate_domain_host<IterateDomainBase, LocalDomain> > > {};

}
