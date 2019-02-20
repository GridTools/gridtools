! GridTools
!
! Copyright (c) 2019, ETH Zurich
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause

module gt_handle
    implicit none
    interface
        subroutine gt_release(h) bind(c)
            use iso_c_binding
            type(c_ptr), value :: h
        end
    end interface
end
