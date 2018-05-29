#pragma once

#include "./interface/layout_transformation/layout_transformation.hpp"
#include "../c_bindings/fortran_array_view.hpp"
#include "../storage/common/storage_info_rt.hpp"

namespace gridtools {
    namespace impl {
        template < class T >
        struct view_info {
            T *fortran_pointer;
            T *cpp_pointer;
            std::vector< uint_t > dims;
            std::vector< uint_t > fortran_strides;
            std::vector< uint_t > cpp_strides;
        };
        template < class DataStore,
            class Layout = typename DataStore::storage_info_t::layout_t,
            class ElementType = typename DataStore::data_t >
        view_info< ElementType > get_view_info(
            const DataStore &dataStore, const gt_fortran_array_descriptor &descriptor) {

            view_info< ElementType > result;

            storage_info_rt si = make_storage_info_rt(*dataStore.get_storage_info_ptr());
            result.dims = si.dims();
            result.cpp_strides = si.strides();
            result.fortran_pointer = reinterpret_cast< ElementType * >(descriptor.data);
            result.cpp_pointer = make_host_view(dataStore).data();

            // verify dimensions of fortran array
            for (uint_t c_dim = 0, fortran_dim = 0; c_dim < Layout::masked_length; ++c_dim) {
                if (Layout::at(c_dim) >= 0) {
                    if (descriptor.dims[fortran_dim] != result.dims[c_dim])
                        throw std::runtime_error("dimensions do not match");
                    ++fortran_dim;
                }
            }

            uint_t current_stride = 1;
            for (uint_t i = 0; i < Layout::masked_length; ++i) {
                if (Layout::at(i) >= 0) {
                    result.fortran_strides.push_back(current_stride);
                    current_stride *= result.dims[i];
                } else {
                    result.fortran_strides.push_back(0);
                }
            }

            return result;
        }
    }

    template < class DataStore,
        class StorageInfo = typename DataStore::storage_info_t,
        class Layout = typename DataStore::storage_info_t::layout_t,
        class = enable_if_t< is_data_store< remove_const_t< DataStore > >::value > >
    class View {
      public:
        View(const gt_fortran_array_descriptor &descriptor) : descriptor_(descriptor) {
            if (descriptor_.rank != gt_view_rank::value)
                throw std::runtime_error("rank does not match");
        }

        View(const View &) = delete;
        View(View &&other) = default;

        using gt_view_rank = std::integral_constant< size_t, Layout::unmasked_length >;
        using gt_view_element_type = typename DataStore::data_t;

        const gt_fortran_array_descriptor &descriptor() const { return descriptor_; }

      private:
        gt_fortran_array_descriptor descriptor_;
    };
    template < class DataStore, class >
    void transform(View< DataStore > &dest, const DataStore &src) {
        if (!dest.descriptor_.data)
            throw std::runtime_error("no array to assign to");

        auto view_info = impl::get_view_info(src, dest.descriptor());

        interface::transform(view_info.fortran_pointer,
            view_info.cpp_pointer,
            view_info.dims,
            view_info.fortran_strides,
            view_info.cpp_strides);
    }
    template < class DataStore, class = enable_if_t< is_data_store< remove_const_t< DataStore > >::value > >
    void transform(DataStore &dest, const View< DataStore > &src) {
        auto view_info = impl::get_view_info(dest, src.descriptor());

        interface::transform(view_info.cpp_pointer,
            view_info.fortran_pointer,
            view_info.dims,
            view_info.cpp_strides,
            view_info.fortran_strides);
    }
}
