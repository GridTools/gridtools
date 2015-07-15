#pragma once

#include "../iterate_domain.hpp"
#include "../iterate_domain_metafunctions.hpp"

namespace gridtools {

/**
 * @brief iterate domain class for the CUDA backend
 */
template<template<class> class IterateDomainBase, typename LocalDomain>
class iterate_domain_cuda : public IterateDomainBase<iterate_domain_cuda<IterateDomainBase, LocalDomain> > //CRTP
{
    DISALLOW_COPY_AND_ASSIGN(iterate_domain_cuda);
    typedef IterateDomainBase<iterate_domain_cuda<IterateDomainBase, LocalDomain> > super;
    typedef typename super::data_pointer_array_t data_pointer_array_t;
    typedef typename super::strides_cached_t strides_cached_t;

private:
    const uint_t m_block_size_i;
    const uint_t m_block_size_j;
    data_pointer_array_t* RESTRICT m_data_pointer;
    strides_cached_t* RESTRICT m_strides;

public:
    GT_FUNCTION
    explicit iterate_domain_cuda(LocalDomain const& local_domain, const uint_t block_size_i, const uint_t block_size_j)
        : super(local_domain), m_block_size_i(block_size_i), m_block_size_j(block_size_j), m_data_pointer(0), m_strides(0) {}

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

    /**
     * @brief determines whether the current (i,j) position is within the block size
     */
    GT_FUNCTION
    bool is_thread_in_domain() const
    {
        return threadIdx.x < m_block_size_i && threadIdx.y < m_block_size_j ;
    }

    /**
     * @brief determines whether the current (i,j) position + an offset is within the block size
     */
    GT_FUNCTION
    bool is_thread_in_domain(const int_t i_offset, const int_t j_offset) const
    {
        return is_thread_in_domain_x(i_offset) &&  is_thread_in_domain_y(j_offset);
    }

    /**
     * @brief determines whether the current (i) position is within the block size
     */
    GT_FUNCTION
    bool is_thread_in_domain_x() const
    {
        return threadIdx.x < m_block_size_i;
    }

    /**
     * @brief determines whether the current (i) position + an offset is within the block size
     */
    GT_FUNCTION
    bool is_thread_in_domain_x(const int_t i_offset) const
    {
        return (int_t)threadIdx.x + i_offset >= 0 && (int_t)threadIdx.x + i_offset < m_block_size_i;
    }

    /**
     * @brief determines whether the current (j) position is within the block size
     */
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

    void set_data_pointer_impl(data_pointer_array_t* RESTRICT data_pointer)
    {
        m_data_pointer = data_pointer;
    }

    data_pointer_array_t* RESTRICT data_pointer_impl() const
    {
        return m_data_pointer;
    }
    strides_cached_t* RESTRICT strides_impl() const
    {
        return m_strides;
    }
    void set_strides_impl(strides_cached_t* RESTRICT strides)
    {
        m_strides = strides;
    }
};

template<
    template<class> class IterateDomainBase, typename LocalDomain>
struct is_iterate_domain<
    iterate_domain_cuda<IterateDomainBase, LocalDomain>
> : public boost::mpl::true_{};

template<
    template<class> class IterateDomainBase,
    typename LocalDomain
>
struct is_positional_iterate_domain<iterate_domain_cuda<IterateDomainBase, LocalDomain> > :
    is_positional_iterate_domain<IterateDomainBase<iterate_domain_cuda<IterateDomainBase, LocalDomain> > > {};

}
