#pragma once

#include <gridtools.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/mpl/has_key.hpp>
#include "../level.hpp"

#include "backend_traits_cuda.hpp"
#include "../mss_functor.hpp"
#include "../sfinae.hpp"
#include "../../storage/meta_storage.hpp"

namespace gridtools{

    template<enumtype::strategy>
    struct strategy_from_id_cuda;

    /**
       @brief specialization for the \ref gridtools::_impl::Naive strategy
    */
    template<>
    struct strategy_from_id_cuda< enumtype::Naive>
    {
    };

    /**
       @brief specialization for the \ref gridtools::_impl::Block strategy
       Empty as not used in the CUDA backend
    */
    template<>
    struct strategy_from_id_cuda <enumtype::Block> {
        // default block size for Block strategy
        typedef block_size<GT_DEFAULT_TILE_I, GT_DEFAULT_TILE_J> block_size_t;

        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponentsArray a meta array with the mss components of all MSS
         * @tparam BackendId id of the backend
         */
        template<typename MssComponentsArray, enumtype::backend BackendId>
        struct fused_mss_loop
        {
            GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssComponentsArray, is_mss_components>::value), "Internal Error: wrong type");
            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename MssComponentsArray::elements>::type::value> iter_range;

            template<typename LocalDomainListArray, typename Coords>
            static void run(LocalDomainListArray& local_domain_lists, const Coords& coords)
            {
                GRIDTOOLS_STATIC_ASSERT((is_coordinates<Coords>::value), "Internal Error: wrong type");

                typedef backend_traits_from_id< BackendId > backend_traits;
                gridtools::for_each<iter_range> (
                    mss_functor<MssComponentsArray, Coords, LocalDomainListArray, BackendId, enumtype::Block> (local_domain_lists, coords,0,0)
                );
            }
        };


        template <typename Index, typename Layout, typename ... Tiles>
        struct get_tmp_storage_info
        {
            typedef storage_info<Index::value, Layout, true, Tiles ...> type;
        };

        /**
         * @brief metafunction that returns the storage type for the storage type of the temporaries for this strategy.
         * with the naive algorithms, the temporary storages are like the non temporary ones
         */
        template <typename Storage, typename ... Tiles>
        struct get_tmp_storage
        {
            typedef storage<base_storage<typename Storage::pointer_type, typename get_tmp_storage_info<typename Storage::meta_data_t::index_type, typename Storage::meta_data_t::layout, Tiles ...
                                                                                                       >::type, Storage::field_dimensions > > type;
        };
    };

}//namespace gridtools
