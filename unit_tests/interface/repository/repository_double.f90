! This file is generated!
module repository
implicit none
  interface

    type(c_ptr) function make_exported_repository(arg0, arg1, arg2) bind(c)
      use iso_c_binding
      integer(c_int), value :: arg0
      integer(c_int), value :: arg1
      integer(c_int), value :: arg2
    end function
    subroutine prefix_set_exported_ijfield_impl(arg0, arg1) bind(c, name="prefix_set_exported_ijfield")
      use iso_c_binding
      use gen_array_descriptor
      type(c_ptr), value :: arg0
      type(gen_fortran_array_descriptor) :: arg1
    end subroutine
    subroutine prefix_set_exported_ijkfield_impl(arg0, arg1) bind(c, name="prefix_set_exported_ijkfield")
      use iso_c_binding
      use gen_array_descriptor
      type(c_ptr), value :: arg0
      type(gen_fortran_array_descriptor) :: arg1
    end subroutine
    subroutine prefix_set_exported_jkfield_impl(arg0, arg1) bind(c, name="prefix_set_exported_jkfield")
      use iso_c_binding
      use gen_array_descriptor
      type(c_ptr), value :: arg0
      type(gen_fortran_array_descriptor) :: arg1
    end subroutine
    subroutine verify_exported_repository(arg0) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
    end subroutine

  end interface
contains
    subroutine prefix_set_exported_ijfield(arg0, arg1)
      use iso_c_binding
      use gen_array_descriptor
      type(c_ptr), value, target :: arg0
      real(c_double), dimension(:,:), target :: arg1
      type(gen_fortran_array_descriptor) :: descriptor1

      !$acc data present(arg1)
      !$acc host_data use_device(arg1)
      descriptor1%rank = 2
      descriptor1%type = 6
      descriptor1%dims = reshape(shape(arg1), &
        shape(descriptor1%dims), (/0/))
      descriptor1%data = c_loc(arg1(lbound(arg1, 1),lbound(arg1, 2)))
      !$acc end host_data
      !$acc end data

      call prefix_set_exported_ijfield_impl(arg0, descriptor1)
    end subroutine
    subroutine prefix_set_exported_ijkfield(arg0, arg1)
      use iso_c_binding
      use gen_array_descriptor
      type(c_ptr), value, target :: arg0
      real(c_double), dimension(:,:,:), target :: arg1
      type(gen_fortran_array_descriptor) :: descriptor1

      !$acc data present(arg1)
      !$acc host_data use_device(arg1)
      descriptor1%rank = 3
      descriptor1%type = 6
      descriptor1%dims = reshape(shape(arg1), &
        shape(descriptor1%dims), (/0/))
      descriptor1%data = c_loc(arg1(lbound(arg1, 1),lbound(arg1, 2),lbound(arg1, 3)))
      !$acc end host_data
      !$acc end data

      call prefix_set_exported_ijkfield_impl(arg0, descriptor1)
    end subroutine
    subroutine prefix_set_exported_jkfield(arg0, arg1)
      use iso_c_binding
      use gen_array_descriptor
      type(c_ptr), value, target :: arg0
      real(c_double), dimension(:,:), target :: arg1
      type(gen_fortran_array_descriptor) :: descriptor1

      !$acc data present(arg1)
      !$acc host_data use_device(arg1)
      descriptor1%rank = 2
      descriptor1%type = 6
      descriptor1%dims = reshape(shape(arg1), &
        shape(descriptor1%dims), (/0/))
      descriptor1%data = c_loc(arg1(lbound(arg1, 1),lbound(arg1, 2)))
      !$acc end host_data
      !$acc end data

      call prefix_set_exported_jkfield_impl(arg0, descriptor1)
    end subroutine
end
