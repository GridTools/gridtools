#pragma once

#include "../iterate_domain.h"
#include "../iterate_domain_metafunctions.h"

namespace gridtools {

template<template<class> class IterateDomainBase, typename LocalDomain>
class iterate_domain_cuda : public IterateDomainBase<iterate_domain_cuda<IterateDomainBase, LocalDomain> > //CRTP
{
    DISALLOW_COPY_AND_ASSIGN(iterate_domain_cuda);
    typedef IterateDomainBase<iterate_domain_cuda<IterateDomainBase, LocalDomain> > super;
public:
    GT_FUNCTION
    explicit iterate_domain_cuda(LocalDomain const& local_domain, const uint_t block_size_i, const uint_t block_size_j)
        : super(local_domain), m_block_size_i(block_size_i), m_block_size_j(block_size_j) {}

    GT_FUNCTION
    uint_t thread_position_x() const
    {
        //TODOCOSUNA implement an acc assert
//        assert(idx < 2);
        return threadIdx.x;
    }

    GT_FUNCTION
    uint_t thread_position_y() const
    {
        //TODOCOSUNA implement an acc assert
//        assert(idx < 2);
        return threadIdx.y;
    }

    GT_FUNCTION
    bool is_thread_in_domain() const
    {
        return threadIdx.x < m_block_size_i && threadIdx.y < m_block_size_j ;
    }

    GT_FUNCTION
    bool is_thread_in_domain(const int_t i_offset, const int_t j_offset) const
    {
        return (int_t)threadIdx.x + i_offset >= 0 && (int_t)threadIdx.x + i_offset < m_block_size_i &&
            (int_t)threadIdx.y + j_offset >= 0 && (int_t)threadIdx.y + j_offset < m_block_size_j ;
    }

    GT_FUNCTION
    bool is_thread_in_domain_x() const
    {
        return threadIdx.x < m_block_size_i;
    }

    GT_FUNCTION
    bool is_thread_in_domain_x(const int_t i_offset) const
    {
        return (int_t)threadIdx.x + i_offset >= 0 && (int_t)threadIdx.x + i_offset < m_block_size_i;
    }

    GT_FUNCTION
    bool is_thread_in_domain_y(const int_t j_offset) const
    {
        return (int_t)threadIdx.y + j_offset >= 0 && (int_t)threadIdx.y + j_offset < m_block_size_j;
    }

    GT_FUNCTION
    uint block_size_i()
    {
        return m_block_size_i;
    }
    GT_FUNCTION
    uint block_size_j()
    {
        return m_block_size_j;
    }

private:
    //TODOCOSUNA use gridtools array
    const uint_t m_block_size_i;
    const uint_t m_block_size_j;
//    uint_t m_thread_position[2];
};

//template<typename T> struct is_iterate_domain;

template<
    template<class> class IterateDomainBase, typename LocalDomain>
struct is_iterate_domain<
    iterate_domain_cuda<IterateDomainBase, LocalDomain>
> :
    public boost::mpl::true_{};

template<
    template<class> class IterateDomainBase,
    typename LocalDomain
>
struct is_positional_iterate_domain<iterate_domain_cuda<IterateDomainBase, LocalDomain> > :
    is_positional_iterate_domain<IterateDomainBase<iterate_domain_cuda<IterateDomainBase, LocalDomain> > > {};

}
