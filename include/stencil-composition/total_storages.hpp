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

namespace gridtools {

    /**@brief metafunction that counts the total number of data fields which are neceassary for this functor (i.e.
     * number of storage
     * instances times number of fields per storage)
    */
    template < typename StorageWrapperList, int_t EndIndex >
    struct total_storages {

        DISALLOW_COPY_AND_ASSIGN(total_storages);
        // the index must not exceed the number of storages
        GRIDTOOLS_STATIC_ASSERT(EndIndex <= boost::fusion::result_of::size< StorageWrapperList >::type::value,
            "the index must not exceed the number of storages");

        typedef
            typename boost::mpl::if_c< (EndIndex < 0),
                boost::mpl::vector0<>,
                typename boost::mpl::fold< typename reversed_range< uint_t, 0, EndIndex >::type,
                                           boost::mpl::vector0<>,
                                           boost::mpl::push_back< boost::mpl::_1,
                                               boost::mpl::at< StorageWrapperList, boost::mpl::_2 > > >::type >::type
                storages_wrappers_t;

        typedef typename boost::mpl::fold< storages_wrappers_t,
            boost::mpl::int_< 0 >,
            boost::mpl::plus< boost::mpl::_1, storage_size_from_storage_wrapper< boost::mpl::_2 > > >::type type;

        static const uint_t value = type::value;

      private:
        total_storages();
    };
} // namespace gridtools
