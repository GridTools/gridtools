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

#if defined(USE_SERIALBOX)

#include "gtest/gtest.h"

#include "stencil-composition/stencil-composition.hpp"
#include "tools/verifier.hpp"

#include <serialbox/gridtools/serialbox.hpp>
#include <unordered_map>
#include <memory>

using namespace gridtools;
using namespace enumtype;

class serialization_setup : public ::testing::Test {
  private:
    std::string m_prefix;
    std::string m_directory;
    std::unordered_map< std::string, storage_t > m_storage_map;
    std::shared_ptr< verifier > m_verifier;

  public:
    using backend_t = backend< Host, structured, Naive >;
    using storage_info_t = backend_t::storage_traits_t::storage_info_t< 0, 3 >;
    using storage_t = backend_t::storage_traits_t::data_store_t< float_type, storage_info_t >;

    /**
     * @brief Allocate the storage `name` of dimensions `dims`
     */
    template < typename Initializer, typename... Dims >
    storage_t make_storage(std::string name, Initializer &&F, Dims &&... dims) {
        storage_info_t storage_info(dims...);
        storage_t storage(storage_info, F, name.c_str());
        m_storage_map[name] = storage;
        return storage;
    }

    /**
     * @brief Get directory of the serializer
     */
    const std::string &directory() const { return m_directory; }

    /**
     * @brief Get prefix of the serializer
     */
    const std::string &prefix() const { return m_prefix; }

    /**
     * @brief Check if all registered fields are present in the serializer
     */
    ::testing::AssertionResult verify_field_meta_info(serialbox::gridtools::serializer &serializer) const {
        for (auto it = m_storage_map.begin(), end = m_storage_map.end(); it != end; ++it) {
            if (!serializer.has_field(it->first))
                return ::testing::AssertionFailure() << "storage " << it->first << " is not present";

            const auto &info = serializer.get_field_meta_info(it->first);
            const auto &dims_array = to_vector(make_unaligned_dims_array(*it->second.get_storage_info_ptr()));
            std::vector< int > dims(&dims_array[0], &dims_array[0] + dims_array.size());

            if (dims != info.dims())
                return ::testing::AssertionFailure() << "dimension mismatch of storage" << it->first
                                                     << "\nRegistered: " << serialbox::ArrayUtil::toString(info.dims())
                                                     << "\nGiven     : " << serialbox::ArrayUtil::toString(dims);
        }
        return ::testing::AssertionSuccess();
    }

    /**
     * @brief Verify storages
     */
    template < class StorageType1, class StorageType2, class GridType >
    ::testing::AssertionResult verify_storages(
        StorageType1 &&storage1, StorageType2 &&storage2, GridType &&grid, uint_t halo_size = 0) const {
        array< array< uint_t, 2 >, 3 > verifier_halos{
            {{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}};
        if (!m_verifier->verify(grid, storage1, storage2, verifier_halos))
            return ::testing::AssertionFailure() << "errors in storage: " << storage1.name();
        return ::testing::AssertionSuccess();
    }

    serialization_setup() {
        const ::testing::TestInfo *testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
        assert(testInfo);

        m_directory = std::string("./") + testInfo->test_case_name() + "/" + testInfo->name();
        m_prefix = "stencil";

#if FLOAT_PRECISION == 4
        m_verifier = std::make_shared< verifier >(1e-6);
#else
        m_verifier = std::make_shared< verifier >(1e-12);
#endif
    }
};

#endif
