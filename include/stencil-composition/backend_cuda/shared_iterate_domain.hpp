/**@file
   @brief file with classes to store the data members of the iterate domain
   that will be allocated in shared memory
 */
#pragma once
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/at.hpp>
#include <boost/fusion/sequence/intrinsic/at_key.hpp>
#include <boost/fusion/include/at_key.hpp>
#include "stencil-composition/accessor.hpp"
#include "common/generic_metafunctions/fusion_map_to_mpl_map.hpp"

namespace gridtools {

    /**
     * @class shared_iterate_domain
     * data structure that holds data members of the iterate domain that must be stored in shared memory.
     * @tparam DataPointerArray array of data pointers
     * @tparam StridesType strides cached type
     * @tparam IJCachesTuple fusion map of <index_type, cache_storage>
     */
    template < typename DataPointerArray, typename StridesType, typename MaxExtent, typename IJCachesTuple >
    class shared_iterate_domain {
        GRIDTOOLS_STATIC_ASSERT((is_strides_cached< StridesType >::value), "Internal Error: wrong type");
        DISALLOW_COPY_AND_ASSIGN(shared_iterate_domain);

      private:
        DataPointerArray m_data_pointer;
        StridesType m_strides;
        IJCachesTuple m_ij_caches_tuple;

        // For some reasons fusion metafunctions (such as result_of::at_key) fail on a fusion map
        // constructed with the result_of::as_map from a fusion vector.
        // Therefore we construct here a mirror metadata mpl map type to be used for meta algorithms
        typedef typename fusion_map_to_mpl_map< IJCachesTuple >::type ij_caches_map_t;

      public:
        shared_iterate_domain() {}

        GT_FUNCTION
        DataPointerArray const &data_pointer() const { return m_data_pointer; }
        GT_FUNCTION
        StridesType const &strides() const { return m_strides; }
        GT_FUNCTION
        DataPointerArray &data_pointer() { return m_data_pointer; }
        GT_FUNCTION
        StridesType &strides() { return m_strides; }

        template < typename IndexType >
        GT_FUNCTION typename boost::mpl::at< ij_caches_map_t, IndexType >::type &RESTRICT get_ij_cache() {
            GRIDTOOLS_STATIC_ASSERT(
                (boost::mpl::has_key< ij_caches_map_t, IndexType >::value), "Accessing a non registered cached");
            return boost::fusion::at_key< IndexType >(m_ij_caches_tuple);
        }
    };

} // namespace gridtools
