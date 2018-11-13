! This file is generated!
module implementation
implicit none
  interface

    type(c_ptr) function create_copy_stencil(arg0, arg1) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      type(c_ptr), value :: arg1
    end function
    type(c_ptr) function create_data_store(arg0, arg1, arg2, arg3) bind(c)
      use iso_c_binding
      integer(c_int), value :: arg0
      integer(c_int), value :: arg1
      integer(c_int), value :: arg2
      real(c_double), dimension(*) :: arg3
    end function
    type(c_ptr) function generic_create_data_store0(arg0, arg1, arg2, arg3) bind(c)
      use iso_c_binding
      integer(c_int), value :: arg0
      integer(c_int), value :: arg1
      integer(c_int), value :: arg2
      real(c_double), dimension(*) :: arg3
    end function
    type(c_ptr) function generic_create_data_store1(arg0, arg1, arg2, arg3) bind(c)
      use iso_c_binding
      integer(c_int), value :: arg0
      integer(c_int), value :: arg1
      integer(c_int), value :: arg2
      real(c_float), dimension(*) :: arg3
    end function
    subroutine run_stencil(arg0) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
    end subroutine
    subroutine sync_data_store(arg0) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
    end subroutine

  end interface
  interface generic_create_data_store
    procedure generic_create_data_store0, generic_create_data_store1
  end interface
contains
end
