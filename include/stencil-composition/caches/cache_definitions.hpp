#pragma once
/**
   @file
   @brief File containing all definitions and enums required by the cache implementations
*/

namespace gridtools {
/**
* @enum CacheIOPolicy
* Enum listing the cache IO policies
*/
enum CacheIOPolicy
{
    cFillAndFlush,  /**< Read values from the cached field and write the result back */
    cFill,          /**< Read values form the cached field but do not write back */
    cFlush,         /**< Write values back the the cached field but do not read in */
    cLocal          /**< Local only cache, neither read nor write the the cached field */
};

/**
 * @enum CacheType
 * enum with the different types of cache available
 */
enum CacheType
{
    IJ,  // IJ caches require synchronization capabilities, as different (i,j) grid points are
         // processed by parallel cores. GPU backend keeps them in shared memory
    K,   // processing of all the K elements is done by same thread, so resources for K caches can be private
         // and do not require synchronization. GPU backend uses registers.
    IJK, // IJK caches is an extension to 3rd dimension of IJ caches. GPU backend uses shared memory
};

}
