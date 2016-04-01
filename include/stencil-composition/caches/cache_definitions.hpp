#pragma once
/**
   @file
   @brief File containing all definitions and enums required by the cache implementations
*/

namespace gridtools {
    /**
    * @enum cache_io_policy
    * Enum listing the cache IO policies
    */
    enum cache_io_policy {
        fill_and_flush, /**< Read values from the cached field and write the result back */
        fill,           /**< Read values form the cached field but do not write back */
        flush,          /**< Write values back the the cached field but do not read in */
        local           /**< Local only cache, neither read nor write the the cached field */
    };

    /**
     * @enum cache_type
     * enum with the different types of cache available
     */
    enum cache_type {
        IJ,    // IJ caches require synchronization capabilities, as different (i,j) grid points are
               // processed by parallel cores. GPU backend keeps them in shared memory
        K,     // processing of all the K elements is done by same thread, so resources for K caches can be private
               // and do not require synchronization. GPU backend uses registers.
        IJK,   // IJK caches is an extension to 3rd dimension of IJ caches. GPU backend uses shared memory
        bypass // bypass the cache for read only parameters
    };
}
