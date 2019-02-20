! GridTools
!
! Copyright (c) 2019, ETH Zurich
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause

module array_descriptor
    use iso_c_binding
    implicit none

    type, bind(c), public :: gt_fortran_array_descriptor
        integer(c_int) :: type
        integer(c_int) :: rank
        integer(c_int), dimension(7) :: dims
        type(c_ptr) :: data
        ! TODO: add support for strides, bounds end type gt_fortran_array_descriptor
    end type gt_fortran_array_descriptor
end module
