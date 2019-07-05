! GridTools
!
! Copyright (c) 2014-2019, ETH Zurich
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause

program main
    use iso_c_binding
    use bindgen_handle
    use implementation
    implicit none
    integer, parameter :: i = 9, j = 10, k = 11
    real(GT_FLOAT_PRECISION), dimension(i, j, k) :: in, out
    type(c_ptr) in_handle, out_handle, stencil

    in = initial()

    in_handle = generic_create_data_store(i, j, k, in(:,1,1))
    out_handle = generic_create_data_store(i, j, k, out(:,1,1))
    stencil = create_copy_stencil(in_handle, out_handle)

    call run_stencil(stencil)
    call sync_data_store(in_handle)
    call sync_data_store(out_handle)

    if (any(in /= initial())) stop 1
    if (any(out /= initial())) stop 1

    call bindgen_release(stencil)
    call bindgen_release(out_handle)
    call bindgen_release(in_handle)

    print *, "It works!"

contains
    function initial()
        integer :: x
        integer, dimension(i, j, k) :: initial
        initial = reshape((/(x, x = 1, size(initial))/) , shape(initial))
    end
end
