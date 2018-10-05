#pragma once

#include "../c_bindings/fortran_array_view.hpp"
#include "../storage/common/storage_info_rt.hpp"
#include "./layout_transformation/layout_transformation.hpp"

namespace gridtools {
    template <class DataStore,
        class StorageInfo = typename DataStore::storage_info_t,
        class Layout = typename DataStore::storage_info_t::layout_t>
    class fortran_array_adapter {
        static_assert(is_data_store<remove_const_t<DataStore>>::value, "");

      public:
        fortran_array_adapter(const gt_fortran_array_descriptor &descriptor) : m_descriptor(descriptor) {
            if (m_descriptor.rank != gt_view_rank::value)
                throw std::runtime_error("rank does not match (descriptor-rank [" + std::to_string(m_descriptor.rank) +
                                         "] != datastore-rank [" + std::to_string(gt_view_rank::value) + "]");
        }

        fortran_array_adapter(const fortran_array_adapter &) = delete;
        fortran_array_adapter(fortran_array_adapter &&other) = default;

        using gt_view_rank = std::integral_constant<size_t, Layout::unmasked_length>;
        using gt_view_element_type = typename DataStore::data_t;
        using gt_is_acc_present = bool_constant<true>;

        friend void transform(DataStore &dest, const fortran_array_adapter &src) {
            adapter{const_cast<fortran_array_adapter &>(src), dest}.from_array();
        }
        friend void transform(fortran_array_adapter &dest, const DataStore &src) {
            adapter{dest, const_cast<DataStore &>(src)}.to_array();
        }

      private:
        class adapter {
            using ElementType = typename DataStore::data_t;

            ElementType *get_ptr_to_first_element(DataStore &data_store) {
#ifdef __CUDACC__
                if (is_cuda_storage<typename DataStore::storage_t>::value) {
                    return make_device_view(data_store).ptr_to_first_position();
                } else {
#endif
                    return make_host_view(data_store).ptr_to_first_position();
#ifdef __CUDACC__
                }
#endif
            }

          public:
            adapter(fortran_array_adapter &view, DataStore &data_store) {

                storage_info_rt si = make_storage_info_rt(*data_store.get_storage_info_ptr());
                m_dims = si.dims();
                m_cpp_strides = si.strides();
                m_fortran_pointer = static_cast<ElementType *>(view.m_descriptor.data);
                m_cpp_pointer = get_ptr_to_first_element(data_store);

                if (!m_fortran_pointer)
                    throw std::runtime_error("No array to assigned to!");

                // verify dimensions of fortran array
                for (uint_t c_dim = 0, fortran_dim = 0; c_dim < Layout::masked_length; ++c_dim) {
                    if (Layout::at(c_dim) >= 0) {
                        if (view.m_descriptor.dims[fortran_dim] != m_dims[c_dim])
                            throw std::runtime_error("dimensions do not match (descriptor [" +
                                                     std::to_string(view.m_descriptor.dims[fortran_dim]) +
                                                     "] != data_store [" + std::to_string(m_dims[c_dim]) + "])");
                        ++fortran_dim;
                    }
                }

                uint_t current_stride = 1;
                for (uint_t i = 0; i < Layout::masked_length; ++i) {
                    if (Layout::at(i) >= 0) {
                        m_fortran_strides.push_back(current_stride);
                        current_stride *= m_dims[i];
                    } else {
                        m_fortran_strides.push_back(0);
                    }
                }
            }

            void from_array() const {
                interface::transform(m_cpp_pointer, m_fortran_pointer, m_dims, m_cpp_strides, m_fortran_strides);
            }
            void to_array() const {
                interface::transform(m_fortran_pointer, m_cpp_pointer, m_dims, m_fortran_strides, m_cpp_strides);
            }

          private:
            ElementType *m_fortran_pointer;
            ElementType *m_cpp_pointer;
            std::vector<uint_t> m_dims;
            std::vector<uint_t> m_fortran_strides;
            std::vector<uint_t> m_cpp_strides;
        };

        const gt_fortran_array_descriptor &m_descriptor;
    };
} // namespace gridtools
