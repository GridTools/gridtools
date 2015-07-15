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
    IJ,
    IJK,
    K
};

}
