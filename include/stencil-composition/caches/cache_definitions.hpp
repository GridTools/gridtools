/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
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
    enum class cache_io_policy {
        fill_and_flush, /**< Read values from the cached field and write the result back */
        fill,           /**< Read values form the cached field but do not write back */
        flush,          /**< Write values back the the cached field but do not read in */
        epflush,        /**< End point cache flush: indicates a flush only at the end point
                              of the interval being cached */
        bpfill,         /**< End point cache fill: indicates a fill only at the begin point
                              of the interval being cached */
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
