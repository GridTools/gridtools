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
#include "../repository/repository.hpp"

#ifndef GRIDTOOLS_REPOSITORY_HAS_VARIANT_WITH_IMPLICIT_CONVERSION
#error "Need repository with implicit conversion activated"
#endif

namespace gridtools {

    namespace _impl {
        class get_storage_info_rt : public boost::static_visitor< storage_info_rt > {
          public:
            template < typename T >
            storage_info_rt operator()(T &t) const {
                return make_storage_info_rt(*t.get_storage_info_ptr());
            }
        };

        class get_pointer_host : public boost::static_visitor< void * > {
          public:
            template < typename T >
            void *operator()(T &t) const {
                return advanced::get_initial_address_of(make_host_view(t));
            }
        };

#ifdef _USE_GPU_
        class get_pointer_cuda : public boost::static_visitor< void * > {
          public:
            template < typename T >
            void *operator()(T &t) const {
                return advanced::get_initial_address_of(make_device_view(t));
            }
        };
#endif
    }

    template < typename Repository >
    class repository_wrapper : public wrappable, public Repository {
      public:
        using Repository::Repository;

        virtual ~repository_wrapper() = default;

        storage_info_rt get_storage_info_rt(const std::string &name, const std::vector< uint_t > &dims) const override {
            assert_has_storage(name);
            storage_info_rt si = boost::apply_visitor(_impl::get_storage_info_rt(), this->data_stores().at(name));
            if (!(si.unaligned_dims() == dims)) {
                LOG(error) << "You are trying to access a data_store where dimensions don't agree.";
            }
            return si;
        }

        void init(const std::string &name, std::vector< uint_t > dims) override {
            assert(false && "repository wrapper assumes all are already initialized");
        }

        void *get_pointer(const std::string &name, storage_type type) override {
            assert_has_storage(name);
            if (type == storage_type::Host) {
                return boost::apply_visitor(_impl::get_pointer_host(), this->data_stores().at(name));
            }
#ifdef _USE_GPU_
            else if (type == storage_type::Cuda) {
                return boost::apply_visitor(_impl::get_pointer_cuda(), this->data_stores().at(name));
            }
#endif
            else {
                throw std::runtime_error(GT_INTERNAL_ERROR_MSG("Storage type is not valid, probably the storage type "
                                                               "is Cuda, when Cuda was disabled on compilation."));
            }
        }

        void notify_push(const std::string &name) override {}
        void notify_pull(const std::string &name) override {}

        bool is_initialized(const std::string &name) override {
            // always initialized by repository constructor
            return true;
        }

        void init_external_pointer(const std::string &name, void *ptr) override {
            LOG_BEGIN("simple_wrapper::set_external_pointer()")
            assert(true && "external pointer mode for repository wrapper not implemented");
            LOG_END()
        }

      private:
        bool has_storage(const std::string &name) const {
            return this->data_stores().find(name) != this->data_stores().end();
        }

        void assert_has_storage(const std::string &name) const {
            if (!has_storage(name))
                throw std::runtime_error("Repository does not contain storage: " + name);
        }
    };
}
