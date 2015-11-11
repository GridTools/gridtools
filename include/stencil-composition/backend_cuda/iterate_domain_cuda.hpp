#pragma once

#include "../iterate_domain.hpp"
#include "../iterate_domain_metafunctions.hpp"
#include "iterate_domain_cache.hpp"
#include "shared_iterate_domain.hpp"

namespace gridtools {

/**
 * @brief iterate domain class for the CUDA backend
 */
template<typename DataPointerArray, typename StridesCached, typename IterateDomainCache, typename IterateDomainArguments>
class iterate_domain_cuda
{
    DISALLOW_COPY_AND_ASSIGN(iterate_domain_cuda);
    GRIDTOOLS_STATIC_ASSERT((is_strides_cached<StridesCached>::value), "Internal error: wrong type");
    GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments<IterateDomainArguments>::value), "Internal error: wrong type");

    typedef IterateDomainBase<iterate_domain_cuda<IterateDomainBase, IterateDomainArguments> > super;
    typedef typename IterateDomainArguments::local_domain_t local_domain_t;
public:

    typedef typename super::data_pointer_array_t data_pointer_array_t;
    typedef typename super::strides_cached_t strides_cached_t;
private:

    typedef shared_iterate_domain<DataPointerArray, StridesCached, typename IterateDomainCache::ij_caches_tuple_t>
        shared_iterate_domain_t;

    typedef typename IterateDomainCache::ij_caches_map_t ij_caches_map_t;

private:
    const uint_t m_block_size_i;
    const uint_t m_block_size_j;

    shared_iterate_domain_t* RESTRICT m_pshared_iterate_domain;

public:
    GT_FUNCTION
    explicit iterate_domain_cuda(const int_t block_size_i, const int_t block_size_j)
        : m_block_size_i(static_cast<uint_t>(block_size_i)), m_block_size_j(static_cast<uint_t>(block_size_j)) {
        //ensure that constructor with default values for block sizes (-1) is not called
        assert(block_size_i > 0 && block_size_j > 0);
    }

    GT_FUNCTION
    uint_t thread_position_x() const
    {
        return threadIdx.x;
    }

    GT_FUNCTION
    uint_t thread_position_y() const
    {
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

    GT_FUNCTION
    void set_shared_iterate_domain_pointer_impl(shared_iterate_domain_t* ptr)
    {
        m_pshared_iterate_domain = ptr;
    }

    GT_FUNCTION
    DataPointerArray const & RESTRICT data_pointer_impl() const
    {
//        assert(m_pshared_iterate_domain);
        return m_pshared_iterate_domain->data_pointer();
    }

    GT_FUNCTION
    DataPointerArray & RESTRICT data_pointer_impl()
    {
//        assert(m_pshared_iterate_domain);
        return m_pshared_iterate_domain->data_pointer();
    }

    GT_FUNCTION
    StridesCached const & RESTRICT strides_impl() const
    {
//        assert((m_pshared_iterate_domain);
        return m_pshared_iterate_domain->strides();
    }
    GT_FUNCTION
    StridesCached & RESTRICT strides_impl()
    {
//        assert((m_pshared_iterate_domain));
        return m_pshared_iterate_domain->strides();
    }

    template <ushort_t Coordinate, typename Execution>
    GT_FUNCTION
    void increment_impl()
    {
        if(Coordinate != 0 && Coordinate != 1) return;
        m_thread_pos[Coordinate] += Execution::value;
    }

    template <ushort_t Coordinate>
    GT_FUNCTION
    void increment_impl(int_t steps)
    {
        if(Coordinate != 0 && Coordinate != 1) return;
        m_thread_pos[Coordinate] += steps;
    }

    template <ushort_t Coordinate>
    GT_FUNCTION
    void initialize_impl()
    {
        if(Coordinate == 0)
            m_thread_pos[Coordinate]=threadIdx.x;
        else if(Coordinate == 1)
            m_thread_pos[Coordinate]=threadIdx.y;
    }

    /** @brief metafunction that determines if an arg is pointing to a field which is read only by all ESFs
    */
    template<typename Accessor>
    struct accessor_points_to_readonly_arg
    {

        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Wrong type");

        typedef typename boost::mpl::at<
            local_domain_args_t, boost::mpl::integral_c<int, Accessor::index_type::value>
        >::type arg_t;

        typedef typename
            boost::mpl::has_key<
                readonly_args_indices_t,
                boost::mpl::integral_c<int, arg_index<arg_t>::value  >
            >::type type;

    };

    /**
    * @brief metafunction that determines if an accessor has to be read from texture memory
    */
    template<typename Accessor>
    struct accessor_read_from_texture
    {
        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Wrong type");
        typedef typename boost::mpl::and_<
            typename accessor_points_to_readonly_arg<Accessor>::type,
            typename boost::mpl::not_< typename boost::mpl::has_key<bypass_caches_set_t, static_uint<Accessor::index_type::value> >::type >::type
        >::type type;
    };

    /** @brief return a value that was cached
    * specialization where cache is not explicitly disabled by user
    */
    template<typename ReturnType, typename Accessor>
    GT_FUNCTION
    typename boost::disable_if<
        boost::mpl::has_key<bypass_caches_set_t, static_uint<Accessor::index_type::value> >,
        ReturnType
    >::type
    get_cache_value_impl(Accessor const & _accessor) const
    {
        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Wrong type");
        //        assert(m_pshared_iterate_domain);
        // retrieve the ij cache from the fusion tuple and access the element required give the current thread position within
        // the block and the offsets of the accessor
        return m_pshared_iterate_domain->template get_ij_cache<static_uint<Accessor::index_type::value> >().at(m_thread_pos, _accessor.offsets());
    }

    /** @brief return a value that was cached
    * specialization where cache is explicitly disabled by user
    */
    template<typename ReturnType, typename Accessor>
    GT_FUNCTION
    typename boost::enable_if<
        boost::mpl::has_key<bypass_caches_set_t, static_uint<Accessor::index_type::value> >,
        ReturnType
    >::type
    get_cache_value_impl(Accessor const & _accessor) const
    {
        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Wrong type");
        return super::template get_value<Accessor, void * RESTRICT> (_accessor,
                    super::template get_data_pointer<Accessor>(_accessor));
    }

    /** @brief return a the value in gmem pointed to by an accessor
    */
    template<
        typename ReturnType,
        typename StoragePointer
    >
    GT_FUNCTION
    ReturnType get_gmem_value(StoragePointer RESTRICT & storage_pointer, const uint_t pointer_offset) const
    {
        return *(storage_pointer+pointer_offset);
    }

    /** @brief return a the value in memory pointed to by an accessor
    * specialization where the accessor points to an arg which is readonly for all the ESFs in all MSSs
    * Value is read via texture system
    */
    template<
        typename ReturnType,
        typename Accessor,
        typename StoragePointer
    >
    GT_FUNCTION
    typename boost::enable_if<
        typename accessor_read_from_texture<Accessor>::type,
        ReturnType
    >::type
    get_value_impl(StoragePointer RESTRICT & storage_pointer, const uint_t pointer_offset) const
    {
        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Wrong type");
#if __CUDA_ARCH__ >= 350
        // on Kepler use ldg to read directly via read only cache
        return __ldg(storage_pointer + pointer_offset);
#else
        return get_gmem_value<ReturnType>(storage_pointer,pointer_offset);
#endif
    }

    /** @brief return a the value in memory pointed to by an accessor
    * specialization where the accessor points to an arg which is not readonly for all the ESFs in all MSSs
    */
    template<
        typename ReturnType,
        typename Accessor,
        typename StoragePointer
    >
    GT_FUNCTION
    typename boost::disable_if<
        typename accessor_read_from_texture<Accessor>::type,
        ReturnType
    >::type
    get_value_impl(StoragePointer RESTRICT & storage_pointer, const uint_t pointer_offset) const
    {
        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Wrong type");
        return get_gmem_value<ReturnType>(storage_pointer,pointer_offset);
    }

private:
    // array storing the (i,j) position of the current thread within the block
    array<int, 2> m_thread_pos;
};

} //namespace gridtools
