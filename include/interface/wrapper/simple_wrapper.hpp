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

#include <interface/logging.h>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <array>

#include "wrappable.hpp"

namespace gridtools {

    /**
     * Simple wrapper for an application with only one field type.
     * The user needs to provide an implementation of the run()
     */
    template < typename DataStoreType >
    class simple_wrapper {};

    template < template < typename, typename > class DataStore, typename Storage, typename StorageInfo >
    class simple_wrapper< DataStore< Storage, StorageInfo > > : public wrappable {
      public:
        using data_store_t = DataStore< Storage, StorageInfo >;
        // in the simple wrapper sizes.size() has to be equal to the number of dimensions of DataStore in the
        // repository
        // based wrapper it should be all the different dims which will be forwarded to the repository ctor
        simple_wrapper(std::vector< uint_t > sizes)
            : m_storage_info(sizes[0], sizes[1], sizes[2]), // TODO fix hard coded size
              m_storage_info_rt(make_storage_info_rt(m_storage_info)) {
            if (data_store_t::storage_info_t::layout_t::masked_length != sizes.size())
                throw std::runtime_error("wrong number of sizes passed in construction");
        }

        virtual ~simple_wrapper() = default;

        storage_info_rt get_storage_info_rt(const std::string &name, const std::vector< uint_t > &dims) const override {
            // TODO maybe: we could actually allow to instantiate one storage_info per storage and allow different
            // dimensions, then the check would be replaced by initializing a storage_info here.
            if (!(m_storage_info_rt.unaligned_dims() == dims)) {
                LOG(error) << "You are trying to access a data_store where dimensions don't agree.";
            }
            return m_storage_info_rt;
        }

        void init(const std::string &name, std::vector< uint_t > dims) override {
            // dims is not used here but would be needed if we would allow different sizes for each storage
            LOG_BEGIN("simple_wrapper::init()")
            if (fields.count(name) == 0) {
                fields.emplace(name, data_store_t(m_storage_info, name));
                LOG(info) << "initialized a new gridtools field";
            } else {
                if (!fields[name].valid()) {
                    fields[name].allocate(m_storage_info);
                    LOG(info) << "allocated a new gridtools field (was uninitialized in map)";
                }
            }
            LOG_END()
        }

        void init_external_pointer(const std::string &name, void *ptr) override {
            LOG_BEGIN("simple_wrapper::set_external_pointer()")
            LOG(info) << "simple_wrapper: updating ptr";

            // TODO initialize if not in map!
            //                if (external_ptr) {
            //                    fields.emplace(name,
            //                        data_store_t(m_storage_info,
            //                                       (float *)ptr,
            //                                       gridtools::ownership::ExternalCPU,
            //                                       name)); // TODO: ownership?, TODO: datatype
            //                    LOG(info) << "initialized a new gridtools field in pointer sharing mode";
            //                } else {
            //                    if (external_ptr) {
            //                        // TODO set ptr
            //                        fields[name].allocate(m_storage_info, (float *)ptr,
            //                        gridtools::ownership::ExternalCPU);
            //                        LOG(info) << "set external ptr (was uninitialized in map)";
            //                    } else {

            if (fields[name].get_storage_ptr()->get_storage_type() == storage_type::Host) {
                assert(false && "implement external ptr mode for a cuda storage");
                //                fields[name].get_storage_ptr()->set_ptrs_impl(static_cast< typename
                //                data_store_t::data_t * >(ptr));
            } else {
                assert(false && "implement external ptr mode for a cuda storage");
                //                fields[name].get_storage_ptr()->set_ptrs_impl(static_cast< typename
                //                data_store_t::data_t * >(ptr));
            }
            LOG_END()
        }

        void *get_pointer(const std::string &name, storage_type type) override {
            if (fields.count(name) > 0) {
                if (type == storage_type::Host) {
                    return advanced::get_initial_address_of(make_host_view(fields[name]));
                }
#ifdef _USE_GPU_
                else {
                    return advanced::get_initial_address_of(make_device_view(fields[name]));
                }
#endif
            }
            return nullptr;
        }

        void notify_push(const std::string &name) override{};
        void notify_pull(const std::string &name) override{};

        bool is_initialized(const std::string &name) override {
            if (fields.count(name) > 0)
                return fields[name].valid(); // in case the field exists but is not yet initialized
            else
                return false;
        };

      protected:
        std::map< std::string, data_store_t > fields;

      private:
        const StorageInfo m_storage_info;
        const storage_info_rt m_storage_info_rt;
    };
}
