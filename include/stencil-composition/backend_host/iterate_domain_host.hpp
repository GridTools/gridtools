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
    typedef typename super::data_pointer_array_t data_pointer_array_t;
    typedef typename super::strides_cached_t strides_cached_t;

public:
    GT_FUNCTION
    explicit iterate_domain_host(LocalDomain const& local_domain)
        : super(local_domain), m_data_pointer(0), m_strides(0) {}

    void set_data_pointer_impl(data_pointer_array_t* RESTRICT data_pointer)
    {
        assert(data_pointer);
        m_data_pointer = data_pointer;
    }

    data_pointer_array_t& RESTRICT data_pointer_impl()
    {
        assert(m_data_pointer);
        return *m_data_pointer;
    }
    data_pointer_array_t const & RESTRICT data_pointer_impl() const
    {
        assert(m_data_pointer);
        return *m_data_pointer;
    }

    strides_cached_t& RESTRICT strides_impl()
    {
        assert(m_strides);
        return *m_strides;
    }

    strides_cached_t const & RESTRICT strides_impl() const
    {
        assert(m_strides);
        return *m_strides;
    }

    void set_strides_pointer_impl(strides_cached_t* RESTRICT strides)
    {
        assert(strides);
        m_strides = strides;
    }

private:
    data_pointer_array_t* RESTRICT m_data_pointer;
    strides_cached_t* RESTRICT m_strides;
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
