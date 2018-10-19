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

#include <gridtools/interface/repository/repository.hpp>
#include <gridtools/storage/storage-facility.hpp>

using IJKStorageInfo = typename gridtools::storage_traits<gridtools::target::x86>::storage_info_t<0, 3>;
using IJKDataStore =
    typename gridtools::storage_traits<gridtools::target::x86>::data_store_t<gridtools::float_type, IJKStorageInfo>;
using IJStorageInfo =
    typename gridtools::storage_traits<gridtools::target::x86>::special_storage_info_t<1, gridtools::selector<1, 1, 0>>;
using IJDataStore =
    typename gridtools::storage_traits<gridtools::target::x86>::data_store_t<gridtools::float_type, IJStorageInfo>;
using JKStorageInfo =
    typename gridtools::storage_traits<gridtools::target::x86>::special_storage_info_t<2, gridtools::selector<0, 1, 1>>;
using JKDataStore =
    typename gridtools::storage_traits<gridtools::target::x86>::data_store_t<gridtools::float_type, JKStorageInfo>;

#define MY_FIELDTYPES (IJKDataStore, (0, 1, 2))(IJDataStore, (0, 1, 2))(JKDataStore, (0, 1, 2))
#define MY_FIELDS (IJKDataStore, ijkfield)(IJDataStore, ijfield)(JKDataStore, jkfield)
GRIDTOOLS_MAKE_REPOSITORY(exported_repository, MY_FIELDTYPES, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS
