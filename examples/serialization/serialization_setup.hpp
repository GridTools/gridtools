/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#if defined(USE_SERIALBOX) && defined(CXX11_ENABLED)

#include "gtest/gtest.h"

#include "stencil-composition/stencil-composition.hpp"
#include "tools/verifier.hpp"

#include <serialbox/gridtools/serialbox.hpp>
#include <unordered_map>
#include <memory>

using namespace gridtools;
using namespace enumtype;

class serialization_setup : public ::testing::Test {
  public:
    typedef layout_map< 0, 1, 2 > layout_t;
    typedef backend< Host, structured, Naive > backend_t;
    typedef backend_t::storage_info< 0, layout_t > meta_data_t;
    typedef backend_t::storage_type< float_type, meta_data_t >::type storage_t;
    typedef backend_t::temporary_storage_type< float_type, meta_data_t >::type temporary_storage_t;

    /**
     * @brief Allocate the storage `name` of dimensions `dims`
     */
    template < typename... Dims >
    storage_t &make_storage(std::string name, Dims &&... dims) {
        meta_data_t meta_data(dims...);
        auto it = m_storage_map.insert({name, std::make_shared< storage_t >(meta_data, name.c_str())}).first;
        return *(it->second.get());
    }

    /**
     * @brief Get storage `name`
     */
    storage_t &get_storage(const std::string &name) const {
        auto it = m_storage_map.find(name);
        if (it == m_storage_map.end())
            throw std::runtime_error("invalid storage \"" + name + "\"");
        return *(it->second.get());
    }

    /**
     * @brief Apply `functor` to each element of the storage `name`
     */
    template < class FunctorType >
    void for_each(const std::string &name, FunctorType &&functor) {
        const auto &storage = get_storage(name);
        const auto &meta_data = storage.meta_data();
        uint_t d1 = meta_data.dim< 0 >();
        uint_t d2 = meta_data.dim< 1 >();
        uint_t d3 = meta_data.dim< 2 >();

        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k)
                    functor(i, j, k);
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
            const auto &dims_array = it->second->meta_data().m_unaligned_dims;
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
            return ::testing::AssertionFailure() << "errors in storage: " << storage1.get_name();
        return ::testing::AssertionSuccess();
    }

    virtual void SetUp() override {
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

    virtual void TearDown() override { m_storage_map.clear(); }

  private:
    std::string m_prefix;
    std::string m_directory;
    std::unordered_map< std::string, std::shared_ptr< storage_t > > m_storage_map;
    std::shared_ptr< verifier > m_verifier;
};

#endif
