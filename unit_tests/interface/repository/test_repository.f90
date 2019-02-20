! GridTools
!
! Copyright (c) 2019, ETH Zurich
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause

subroutine call_repository() bind (c, name="call_repository")

    use iso_c_binding
    use repository

    type(c_ptr) :: repository_handle
#if GT_FLOAT_PRECISION == 4
    integer, parameter :: wp = c_float
#else
    integer, parameter :: wp = c_double
#endif
    real(kind=wp), dimension(3, 4, 5) :: ijkfield
    real(kind=wp), dimension(3, 4) :: ijfield
    real(kind=wp), dimension(4, 5) :: jkfield

    logical(c_bool) :: ret

    ijkfield = reshape( (/ (I, I = 0, 3*4*5) /), shape(ijkfield), (/ 0 /) )
    ijfield = reshape( (/ (I, I = 0, 3*4) /), shape(ijfield), (/ 0 /) )
    jkfield = reshape( (/ (I, I = 0, 4*5) /), shape(jkfield), (/ 0 /) )

    repository_handle = make_exported_repository(3, 4, 5)
    call set_exported_ijkfield(repository_handle, ijkfield)
    call set_exported_ijfield(repository_handle, ijfield)
    call set_exported_jkfield(repository_handle, jkfield)
    call verify_exported_repository(repository_handle)

end subroutine
