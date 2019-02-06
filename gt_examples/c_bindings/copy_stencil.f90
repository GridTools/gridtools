! This file is generated!
module copy_stencil
implicit none
  interface

    type(c_ptr) function make_copy_stencil(arg0) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
    end function
    type(c_ptr) function make_wrapper(arg0, arg1, arg2) bind(c)
      use iso_c_binding
      integer(c_int), value :: arg0
      integer(c_int), value :: arg1
      integer(c_int), value :: arg2
    end function
    subroutine run_stencil_impl(arg0, arg1, arg2, arg3) bind(c, name="run_stencil")
      use iso_c_binding
      use array_descriptor
      type(c_ptr), value :: arg0
      type(c_ptr), value :: arg1
      type(gt_fortran_array_descriptor) :: arg2
      type(gt_fortran_array_descriptor) :: arg3
    end subroutine

  end interface
contains
    subroutine run_stencil(arg0, arg1, arg2, arg3)
      use iso_c_binding
      use array_descriptor
      type(c_ptr), value, target :: arg0
      type(c_ptr), value, target :: arg1
      real(c_float), dimension(:,:,:), target :: arg2
      real(c_float), dimension(:,:,:), target :: arg3
      type(gt_fortran_array_descriptor) :: descriptor2
      type(gt_fortran_array_descriptor) :: descriptor3

      !$acc data present(arg2)
      !$acc host_data use_device(arg2)
      descriptor2%rank = 3
      descriptor2%type = 5
      descriptor2%dims = reshape(shape(arg2), &
        shape(descriptor2%dims), (/0/))
      descriptor2%data = c_loc(arg2(lbound(arg2, 1),lbound(arg2, 2),lbound(arg2, 3)))
      !$acc end host_data
      !$acc end data

      !$acc data present(arg3)
      !$acc host_data use_device(arg3)
      descriptor3%rank = 3
      descriptor3%type = 5
      descriptor3%dims = reshape(shape(arg3), &
        shape(descriptor3%dims), (/0/))
      descriptor3%data = c_loc(arg3(lbound(arg3, 1),lbound(arg3, 2),lbound(arg3, 3)))
      !$acc end host_data
      !$acc end data

      call run_stencil_impl(arg0, arg1, descriptor2, descriptor3)
    end subroutine
end
