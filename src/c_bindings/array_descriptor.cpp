#include "c_bindings/array_descriptor.h"
#include "c_bindings/handle.h"
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

gt_handle *create_data_store_impl(unsigned int, unsigned int, unsigned int, gt_fortran_array_descriptor d) {
    std::cout << d.rank << std::endl;
}
gt_handle *create_data_store_gen_impl1(unsigned int, unsigned int, unsigned int, gt_fortran_array_descriptor) {
    return nullptr;
}
gt_handle *create_data_store_gen_impl2(unsigned int, unsigned int, unsigned int, gt_fortran_array_descriptor) {
    return nullptr;
}

#ifdef __cplusplus
}
#endif
