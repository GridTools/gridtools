    ! GridTools Libraries !
    ! Copyright (c) 2017, ETH Zurich and MeteoSwiss
    ! All rights reserved.
    !
    ! Redistribution and use in source and binary forms, with or without
    ! modification, are permitted provided that the following conditions are
    ! met:
    !
    ! 1. Redistributions of source code must retain the above copyright
    ! notice, this list of conditions and the following disclaimer.
    !
    ! 2. Redistributions in binary form must reproduce the above copyright
    ! notice, this list of conditions and the following disclaimer in the
    ! documentation and/or other materials provided with the distribution.
    !
    ! 3. Neither the name of the copyright holder nor the names of its
    ! contributors may be used to endorse or promote products derived from
    ! this software without specific prior written permission.
    !
    ! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    ! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    ! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    ! A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    ! HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    ! SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    ! LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    ! DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    ! THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    ! (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    ! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    !
    ! For information: http://eth-cscs.github.io/gridtools/

module array_descriptor
    use iso_c_binding
    implicit none

    private
    public :: create_array_descriptor

    type, bind(c), public :: gt_fortran_array_descriptor
        integer(c_int) :: type
        integer(c_int) :: rank
        integer(c_int), dimension(7) :: dims
        type(c_ptr) :: data
        ! TODO: add support for strides, bounds end type gt_fortran_array_descriptor
    end type gt_fortran_array_descriptor

    interface fill_type_info
        procedure fill_type_info1
        procedure fill_type_info2
        procedure fill_type_info3
        procedure fill_type_info4
        procedure fill_type_info5
    end interface

    interface fill_array_dimensions
        procedure fill_array_dimensions1
        procedure fill_array_dimensions2
        procedure fill_array_dimensions3
        procedure fill_array_dimensions4
        procedure fill_array_dimensions5
        procedure fill_array_dimensions6
        procedure fill_array_dimensions7
    end interface

    interface create_array_descriptor
        procedure create_array_descriptor1_1
        procedure create_array_descriptor1_2
        procedure create_array_descriptor1_3
        procedure create_array_descriptor1_4
        procedure create_array_descriptor1_5
        procedure create_array_descriptor2_1
        procedure create_array_descriptor2_2
        procedure create_array_descriptor2_3
        procedure create_array_descriptor2_4
        procedure create_array_descriptor2_5
        procedure create_array_descriptor3_1
        procedure create_array_descriptor3_2
        procedure create_array_descriptor3_3
        procedure create_array_descriptor3_4
        procedure create_array_descriptor3_5
        procedure create_array_descriptor4_1
        procedure create_array_descriptor4_2
        procedure create_array_descriptor4_3
        procedure create_array_descriptor4_4
        procedure create_array_descriptor4_5
        procedure create_array_descriptor5_1
        procedure create_array_descriptor5_2
        procedure create_array_descriptor5_3
        procedure create_array_descriptor5_4
        procedure create_array_descriptor5_5
        procedure create_array_descriptor6_1
        procedure create_array_descriptor6_2
        procedure create_array_descriptor6_3
        procedure create_array_descriptor6_4
        procedure create_array_descriptor6_5
        procedure create_array_descriptor7_1
        procedure create_array_descriptor7_2
        procedure create_array_descriptor7_3
        procedure create_array_descriptor7_4
        procedure create_array_descriptor7_5
    end interface


contains
    subroutine fill_common_info(array, descriptor)
        type(*), dimension(*), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        descriptor%data = c_loc(array)

        print *, "rank: ", descriptor%rank
        print *, "dims: ", descriptor%dims
        print *, "type: ", descriptor%type
    end subroutine
    subroutine fill_type_info1(dummy, descriptor)
        real(c_float), target :: dummy
        type(gt_fortran_array_descriptor) :: descriptor

        descriptor%type = 1
    end subroutine

    subroutine fill_type_info2(dummy, descriptor)
        real(c_double), target :: dummy
        type(gt_fortran_array_descriptor) :: descriptor

        descriptor%type = 2
    end subroutine

    subroutine fill_type_info3(dummy, descriptor)
        integer(c_int), target :: dummy
        type(gt_fortran_array_descriptor) :: descriptor

        descriptor%type = 3
    end subroutine

    subroutine fill_type_info4(dummy, descriptor)
        integer(c_long), target :: dummy
        type(gt_fortran_array_descriptor) :: descriptor

        descriptor%type = 4
    end subroutine

    subroutine fill_type_info5(dummy, descriptor)
        logical(c_bool), target :: dummy
        type(gt_fortran_array_descriptor) :: descriptor

        descriptor%type = 5
    end subroutine

    subroutine fill_array_dimensions1(array, descriptor)
        type(*), dimension(:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        descriptor%rank = 1
        descriptor%dims = reshape(shape(array), &
        shape(descriptor%dims), (/0/))
    end subroutine
    subroutine fill_array_dimensions2(array, descriptor)
        type(*), dimension(:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        descriptor%rank = 2
        descriptor%dims = reshape(shape(array), &
        shape(descriptor%dims), (/0/))
    end subroutine
    subroutine fill_array_dimensions3(array, descriptor)
        type(*), dimension(:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        descriptor%rank = 3
        descriptor%dims = reshape(shape(array), &
        shape(descriptor%dims), (/0/))
    end subroutine
    subroutine fill_array_dimensions4(array, descriptor)
        type(*), dimension(:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        descriptor%rank = 4
        descriptor%dims = reshape(shape(array), &
        shape(descriptor%dims), (/0/))
    end subroutine
    subroutine fill_array_dimensions5(array, descriptor)
        type(*), dimension(:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        descriptor%rank = 5
        descriptor%dims = reshape(shape(array), &
        shape(descriptor%dims), (/0/))
    end subroutine
    subroutine fill_array_dimensions6(array, descriptor)
        type(*), dimension(:,:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        descriptor%rank = 6
        descriptor%dims = reshape(shape(array), &
        shape(descriptor%dims), (/0/))
    end subroutine
    subroutine fill_array_dimensions7(array, descriptor)
        type(*), dimension(:,:,:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        descriptor%rank = 7
        descriptor%dims = reshape(shape(array), &
        shape(descriptor%dims), (/0/))
    end subroutine

    function create_array_descriptor1_1(array) result (descriptor)
        real(c_float), dimension(:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor1_2(array) result (descriptor)
        real(c_double), dimension(:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor1_3(array) result (descriptor)
        integer(c_int), dimension(:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor1_4(array) result (descriptor)
        integer(c_long), dimension(:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor1_5(array) result (descriptor)
        logical(c_bool), dimension(:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor2_1(array) result (descriptor)
        real(c_float), dimension(:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2)), &
            descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor2_2(array) result (descriptor)
        real(c_double), dimension(:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2)), &
            descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor2_3(array) result (descriptor)
        integer(c_int), dimension(:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2)), &
            descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor2_4(array) result (descriptor)
        integer(c_long), dimension(:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2)), &
            descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor2_5(array) result (descriptor)
        logical(c_bool), dimension(:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2)), &
            descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor3_1(array) result (descriptor)
        real(c_float), dimension(:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor3_2(array) result (descriptor)
        real(c_double), dimension(:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor3_3(array) result (descriptor)
        integer(c_int), dimension(:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor3_4(array) result (descriptor)
        integer(c_long), dimension(:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor3_5(array) result (descriptor)
        logical(c_bool), dimension(:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor4_1(array) result (descriptor)
        real(c_float), dimension(:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor4_2(array) result (descriptor)
        real(c_double), dimension(:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor4_3(array) result (descriptor)
        integer(c_int), dimension(:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor4_4(array) result (descriptor)
        integer(c_long), dimension(:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor4_5(array) result (descriptor)
        logical(c_bool), dimension(:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor5_1(array) result (descriptor)
        real(c_float), dimension(:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5)), &
            descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor5_2(array) result (descriptor)
        real(c_double), dimension(:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5)), &
            descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor5_3(array) result (descriptor)
        integer(c_int), dimension(:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5)), &
            descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor5_4(array) result (descriptor)
        integer(c_long), dimension(:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5)), &
            descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor5_5(array) result (descriptor)
        logical(c_bool), dimension(:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5)), &
            descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor6_1(array) result (descriptor)
        real(c_float), dimension(:,:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5), &
            lbound(array, 6)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor6_2(array) result (descriptor)
        real(c_double), dimension(:,:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5), &
            lbound(array, 6)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor6_3(array) result (descriptor)
        integer(c_int), dimension(:,:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5), &
            lbound(array, 6)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor6_4(array) result (descriptor)
        integer(c_long), dimension(:,:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5), &
            lbound(array, 6)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor6_5(array) result (descriptor)
        logical(c_bool), dimension(:,:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5), &
            lbound(array, 6)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor7_1(array) result (descriptor)
        real(c_float), dimension(:,:,:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5), &
            lbound(array, 6), lbound(array, 7)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor7_2(array) result (descriptor)
        real(c_double), dimension(:,:,:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5), &
            lbound(array, 6), lbound(array, 7)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor7_3(array) result (descriptor)
        integer(c_int), dimension(:,:,:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5), &
            lbound(array, 6), lbound(array, 7)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor7_4(array) result (descriptor)
        integer(c_long), dimension(:,:,:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5), &
            lbound(array, 6), lbound(array, 7)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function

    function create_array_descriptor7_5(array) result (descriptor)
        logical(c_bool), dimension(:,:,:,:,:,:,:), target :: array
        type(gt_fortran_array_descriptor) :: descriptor

        call fill_type_info(array(lbound(array, 1), lbound(array, 2), &
            lbound(array, 3), lbound(array, 4), lbound(array, 5), &
            lbound(array, 6), lbound(array, 7)), descriptor)
        call fill_array_dimensions(array, descriptor)
        call fill_common_info(array, descriptor)

    end function
end module
