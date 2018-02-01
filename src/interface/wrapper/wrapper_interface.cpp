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

#include "common/cuda_is_ptr.hpp"
#include "interface/wrapper/wrappable.hpp"
#include "interface/layout_transformation/layout_transformation.hpp"
#include "interface/logging.h"
#include <iostream>
#include <vector>
#include <map>
#include "storage/common/definitions.hpp"
#include "storage/common/storage_info_interface.hpp"
#include "storage/common/storage_info_rt.hpp"
#include "c_bindings/export.hpp"

namespace gridtools {
    namespace impl_ {
        std::vector< gridtools::uint_t > carray_to_vector(int ndims, int *dims) {
            std::vector< gridtools::uint_t > v_dims(ndims);
            for (size_t i = 0; i < ndims; ++i) {
                v_dims[i] = dims[i];
            }
            return v_dims;
        }

        std::array< gridtools::uint_t, 3 > vector_to_array(const std::vector< uint_t > &v) {
            std::array< gridtools::uint_t, 3 > a;
            std::copy(v.begin(), v.end(), a.begin());
            return a;
        }

        // TODO should go into a separate file and capable of doing GPU
        template < typename DataType >
        void copy(DataType *dst,
            DataType *src,
            const std::vector< gridtools::uint_t > &dims,
            const std::vector< uint_t > &dst_strides,
            const std::vector< uint_t > &src_strides) {
            gridtools::interface::transform(dst, src, dims, dst_strides, src_strides);
        }

        bool same_layout(storage_info_rt info, int ndims, int *dims, int *strides) {
            return carray_to_vector(ndims, strides) == info.strides();
        }

        template < typename T >
        storage_type get_storage_type(T *&ptr) {
            return is_gpu_ptr(ptr) ? storage_type::Cuda : storage_type::Host;
        }
    }
}

using namespace gridtools;

template < typename T >
void gt_push_impl(
    std::shared_ptr< wrappable > m, char *name, T *src_ptr, int ndims, int *dims, int *strides, bool force_copy) {

    //    TODO pointer sharing not implemented, remove the following line
    force_copy = 1;

    LOG_BEGIN("wrapper_functions::gt_push_impl()");
    LOG(info) << "push for " << std::string(name) << " with src ptr " << src_ptr;
    LOG(info) << "dims: " << dims[0] << "/" << dims[1] << "/" << dims[2];             // TODO FIXME
    LOG(info) << "strides: " << strides[0] << "/" << strides[1] << "/" << strides[2]; // TODO FIXME

    storage_type type = impl_::get_storage_type(src_ptr);
    auto src_dims = impl_::carray_to_vector(ndims, dims);
    auto src_strides = impl_::carray_to_vector(ndims, strides);

    storage_info_rt dst_info = m->get_storage_info_rt(name, src_dims);

    bool external_ptr_mode = !force_copy && impl_::same_layout(dst_info, ndims, dims, strides);

    if (external_ptr_mode) {
        LOG(info) << "mode: ptr sharing";
        throw std::runtime_error("external ptr mode not implemented");
        // m->init_external_pointer(name, src_ptr); // even if it is already initialized!
    } else {
        if (!m->is_initialized(name)) {
            LOG(info) << name << " is not yet initialized";
            m->init(name, src_dims);
        }

        T *dst_ptr = (T *)m->get_pointer(name, type);

        impl_::copy(
            dst_ptr, src_ptr, dst_info.dims(), dst_info.strides(), src_strides); // TODO or unaligned_dim / strides?s
    }
    m->notify_push(name);
    LOG_END()
}

void gt_push_internal_float(
    std::shared_ptr< wrappable > m, char *name, float *ptr, int ndims, int *dims, int *strides, bool force_copy) {
    gt_push_impl(m, name, ptr, ndims, dims, strides, force_copy);
}
void gt_push_internal_double(
    std::shared_ptr< wrappable > m, char *name, double *ptr, int ndims, int *dims, int *strides, bool force_copy) {
    gt_push_impl(m, name, ptr, ndims, dims, strides, force_copy);
}
void gt_push_internal_int(
    std::shared_ptr< wrappable > m, char *name, int *ptr, int ndims, int *dims, int *strides, bool force_copy) {
    gt_push_impl(m, name, ptr, ndims, dims, strides, force_copy);
}
void gt_push_internal_bool(
    std::shared_ptr< wrappable > m, char *name, bool *ptr, int ndims, int *dims, int *strides, bool force_copy) {
    gt_push_impl(m, name, ptr, ndims, dims, strides, force_copy);
}

GT_EXPORT_BINDING_7(gt_push_float, gt_push_internal_float);
GT_EXPORT_BINDING_7(gt_push_double, gt_push_internal_double);
GT_EXPORT_BINDING_7(gt_push_int, gt_push_internal_int);
GT_EXPORT_BINDING_7(gt_push_bool, gt_push_internal_bool);

template < typename T >
void gt_pull_impl(std::shared_ptr< wrappable > m, char *name, T *ptr, int ndims, int *dims, int *strides) {
    LOG_BEGIN("wrapper_functions::gt_pull_impl()")
    LOG(info) << "pull for " << std::string(name) << " to ptr " << ptr;

    storage_type type = impl_::get_storage_type(ptr);

    T *src_ptr = (T *)m->get_pointer(name, type);
    if (src_ptr == nullptr) {
        throw std::runtime_error("Field is not available for pull");
    } else {
        if (ptr == src_ptr) {
            LOG(info) << "not copying ptr sharing";
        } else {
            gridtools::storage_info_rt src_info = m->get_storage_info_rt(name, impl_::carray_to_vector(ndims, dims));
            impl_::copy(ptr, src_ptr, src_info.dims(), impl_::carray_to_vector(ndims, strides), src_info.strides());
        }
    }
    m->notify_pull(name);
    LOG_END()
}

void gt_pull_internal_float(
    std::shared_ptr< wrappable > m, char *name, float *ptr, int ndims, int *dims, int *strides) {
    gt_pull_impl(m, name, ptr, ndims, dims, strides);
}
void gt_pull_internal_double(
    std::shared_ptr< wrappable > m, char *name, double *ptr, int ndims, int *dims, int *strides) {
    gt_pull_impl(m, name, ptr, ndims, dims, strides);
}
void gt_pull_internal_int(std::shared_ptr< wrappable > m, char *name, int *ptr, int ndims, int *dims, int *strides) {
    gt_pull_impl(m, name, ptr, ndims, dims, strides);
}
void gt_pull_internal_bool(std::shared_ptr< wrappable > m, char *name, bool *ptr, int ndims, int *dims, int *strides) {
    gt_pull_impl(m, name, ptr, ndims, dims, strides);
}

GT_EXPORT_BINDING_6(gt_pull_float, gt_pull_internal_float);
GT_EXPORT_BINDING_6(gt_pull_double, gt_pull_internal_double);
GT_EXPORT_BINDING_6(gt_pull_int, gt_pull_internal_int);
GT_EXPORT_BINDING_6(gt_pull_bool, gt_pull_internal_bool);

/**
 * @brief Getter to obtain the wrapper factories. Due to the static initialization order this
 *        object cannot be stored statically: https://isocpp.org/wiki/faq/ctors#static-init-order
 *
 * @return The map containing the wrapper factories it is a map mapping a string to a function
 *         pointer which acts as a factory.
 */
std::map< std::string, std::function< wrappable *(std::vector< gridtools::uint_t >) > > &get_wrapper_factories() {
    static std::map< std::string, std::function< wrappable *(std::vector< gridtools::uint_t >)> > map;
    return map;
}

void gt_run_impl(std::shared_ptr< wrappable > m) { m->run(); }
GT_EXPORT_BINDING_1(gt_run, gt_run_impl);
