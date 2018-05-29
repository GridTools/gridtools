#pragma once

#include "./interface/layout_transformation/layout_transformation.hpp"
#include "../c_bindings/fortran_array_view.hpp"
#include "../storage/common/storage_info_rt.hpp"

namespace gridtools {
    template < class DataStore,
        class StorageInfo = typename DataStore::storage_info_t,
        class Layout = typename DataStore::storage_info_t::layout_t >
    class View {
        static_assert(is_data_store< remove_const_t< DataStore > >::value, "");

      public:
        View(const gt_fortran_array_descriptor &descriptor) : descriptor_(descriptor) {
            if (descriptor_.rank != gt_view_rank::value)
                throw std::runtime_error("rank does not match (descriptor-rank [" + std::to_string(descriptor_.rank) +
                                         "] != datastore-rank [" + std::to_string(gt_view_rank::value) + "]");
        }

        View(const View &) = delete;
        View(View &&other) = default;

        using gt_view_rank = std::integral_constant< size_t, Layout::unmasked_length >;
        using gt_view_element_type = typename DataStore::data_t;

        friend void transform(DataStore &dest, const View &src) {
            Transformer{const_cast< View & >(src), dest}.fromArray();
        }
        friend void transform(View &dest, const DataStore &src) {
            Transformer{dest, const_cast< DataStore & >(src)}.toArray();
        }

      private:
        class Transformer {
            using ElementType = typename DataStore::data_t;

          public:
            Transformer(View &view, DataStore &dataStore) {

                storage_info_rt si = make_storage_info_rt(*dataStore.get_storage_info_ptr());
                dims = si.dims();
                cpp_strides = si.strides();
                fortran_pointer = static_cast< ElementType * >(view.descriptor_.data);
                cpp_pointer = make_host_view(dataStore).data();

                if (!fortran_pointer)
                    throw std::runtime_error("No array to assigned to!");

                // verify dimensions of fortran array
                for (uint_t c_dim = 0, fortran_dim = 0; c_dim < Layout::masked_length; ++c_dim) {
                    if (Layout::at(c_dim) >= 0) {
                        if (view.descriptor_.dims[fortran_dim] != dims[c_dim])
                            throw std::runtime_error("dimensions do not match (descriptor [" +
                                                     std::to_string(view.descriptor_.dims[fortran_dim]) +
                                                     "] != data_store [" + std::to_string(dims[c_dim]) + "])");
                        ++fortran_dim;
                    }
                }

                uint_t current_stride = 1;
                for (uint_t i = 0; i < Layout::masked_length; ++i) {
                    if (Layout::at(i) >= 0) {
                        fortran_strides.push_back(current_stride);
                        current_stride *= dims[i];
                    } else {
                        fortran_strides.push_back(0);
                    }
                }
            }

            void fromArray() const {
                interface::transform(cpp_pointer, fortran_pointer, dims, cpp_strides, fortran_strides);
            }
            void toArray() const {
                interface::transform(fortran_pointer, cpp_pointer, dims, fortran_strides, cpp_strides);
            }

          private:
            ElementType *fortran_pointer;
            ElementType *cpp_pointer;
            std::vector< uint_t > dims;
            std::vector< uint_t > fortran_strides;
            std::vector< uint_t > cpp_strides;
        };

        const gt_fortran_array_descriptor &descriptor_;
    };
}
