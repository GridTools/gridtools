
module implementation_wrapper
implicit none
  interface

    type(c_ptr) function create_copy_stencil_impl(arg0, arg1) bind(c, name="create_copy_stencil")
      use iso_c_binding
      use array_descriptor
      type(gt_fortran_array_descriptor) :: arg0
      type(gt_fortran_array_descriptor) :: arg1
    end function
    subroutine run_stencil_impl(arg0) bind(c, name="run_stencil")
      use iso_c_binding
      type(c_ptr), value :: arg0
    end subroutine
    subroutine sync_data_store_impl(arg0) bind(c, name="sync_data_store")
      use iso_c_binding
      use array_descriptor
      type(gt_fortran_array_descriptor) :: arg0
    end subroutine

  end interface
contains
    type(c_ptr) function create_copy_stencil(arg0, arg1)
      use iso_c_binding
      use array_descriptor
      real(c_double), dimension(:,:,:), target :: arg0
      real(c_double), dimension(:,:,:), target :: arg1
      type(gt_fortran_array_descriptor) :: descriptor0
      type(gt_fortran_array_descriptor) :: descriptor1

      descriptor0%rank = 3
      descriptor0%type = 6
      descriptor0%dims = reshape(shape(arg0), &
        shape(descriptor0%dims), (/0/))
      descriptor0%data = c_loc(arg0(lbound(arg0, 1),lbound(arg0, 2),lbound(arg0, 3)))

      descriptor1%rank = 3
      descriptor1%type = 6
      descriptor1%dims = reshape(shape(arg1), &
        shape(descriptor1%dims), (/0/))
      descriptor1%data = c_loc(arg1(lbound(arg1, 1),lbound(arg1, 2),lbound(arg1, 3)))

      create_copy_stencil = create_copy_stencil_impl(descriptor0, descriptor1)
    end function
    subroutine run_stencil(arg0)
      use iso_c_binding
      type(c_ptr), value, target :: arg0

      call run_stencil_impl(arg0)
    end subroutine
    subroutine sync_data_store(arg0)
      use iso_c_binding
      use array_descriptor
      real(c_double), dimension(:,:,:), target :: arg0
      type(gt_fortran_array_descriptor) :: descriptor0

      descriptor0%rank = 3
      descriptor0%type = 6
      descriptor0%dims = reshape(shape(arg0), &
        shape(descriptor0%dims), (/0/))
      descriptor0%data = c_loc(arg0(lbound(arg0, 1),lbound(arg0, 2),lbound(arg0, 3)))

      call sync_data_store_impl(descriptor0)
    end subroutine
end
