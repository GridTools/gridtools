! GridTools
!
! Copyright (c) 2014-2019, ETH Zurich
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause

program main
    use iso_c_binding
    use gt_handle
    use copy_stencil_lib_mc
    implicit none
    integer, parameter :: i = 9, j = 10, k = 11
    real(c_float), dimension(i, j, k) :: in_array, out_array
    type(c_ptr) grid_handle, storage_info_handle, computation_handle, in_handle, out_handle

    ! fill some input values
    in_array = initial()
    out_array(:, :, :) = 0

    grid_handle = make_grid(i, j, k)
    storage_info_handle = make_storage_info(i, j, k)
    in_handle = make_data_store(storage_info_handle)
    out_handle = make_data_store(storage_info_handle)
    computation_handle = make_copy_stencil(grid_handle)
    ! gt_handles need to be released explicitly
    call gt_release(grid_handle)
    call gt_release(storage_info_handle)

    ! transform data from Fortran to C layout
    call transform_f_to_c(in_handle, in_array)

    call run_stencil(computation_handle, in_handle, out_handle)

    ! transform data from C layout to Fortran layout
    call transform_c_to_f(out_array, out_handle)

    ! check output
    if (any(in_array /= initial())) stop 1
    if (any(out_array /= initial())) stop 1

    ! gt_handles need to be released explicitly
    call gt_release(in_handle)
    call gt_release(out_handle)
    call gt_release(computation_handle)

    print *, "It works!"

contains
    function initial()
        integer :: x
        integer, dimension(i, j, k) :: initial
        initial = reshape((/(x, x = 1, size(initial))/) , shape(initial))
    end
end
