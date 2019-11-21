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
    use implementation_wrapper
    implicit none
    integer, parameter :: i = 9, j = 10, k = 11
    real(GT_FLOAT_PRECISION), dimension(i, j, k) :: in, out
    type(c_ptr) in_handle, out_handle, stencil
    integer(c_int) :: cnt

    in = initial()

    call run_copy_functor(in, out)

    if (any(in /= initial())) stop 1
    if (any(out /= initial())) stop 1

    print *, "It works!"

contains
    function initial()
        integer :: x
        integer, dimension(i, j, k) :: initial
        initial = reshape((/(x, x = 1, size(initial))/) , shape(initial))
    end
end
