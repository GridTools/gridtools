module descriptor_no_generic
implicit none
  private
  public :: create_data_store

  interface
    type(c_ptr) function create_data_store_impl(arg0, arg1, arg2, arg3) bind(c)
      use iso_c_binding
      use array_descriptor
      integer(c_int), value :: arg0
      integer(c_int), value :: arg1
      integer(c_int), value :: arg2
      type(gt_fortran_array_descriptor), value :: arg3
    end function
  end interface
contains
  function create_data_store(arg0, arg1, arg2, arg3) result(data_store)
    use iso_c_binding
    use array_descriptor
    integer(c_int), value :: arg0
    integer(c_int), value :: arg1
    integer(c_int), value :: arg2
    real(c_double), dimension(:,:,:), target :: arg3
    type(c_ptr) :: data_store

    data_store = create_data_store_impl(arg0, arg1, arg2, create_array_descriptor(arg3))
  end function
end

module descriptor_generic
implicit none
  private
  public :: create_data_store_gen

  interface
    type(c_ptr) function create_data_store_gen_impl1(arg0, arg1, arg2, arg3) bind(c)
      use iso_c_binding
      use array_descriptor
      integer(c_int), value :: arg0
      integer(c_int), value :: arg1
      integer(c_int), value :: arg2
      type(gt_fortran_array_descriptor), value :: arg3
    end function
    type(c_ptr) function create_data_store_gen_impl2(arg0, arg1, arg2, arg3) bind(c)
      use iso_c_binding
      use array_descriptor
      integer(c_int), value :: arg0
      integer(c_int), value :: arg1
      integer(c_int), value :: arg2
      type(gt_fortran_array_descriptor), value :: arg3
    end function
  end interface
  interface create_data_store_gen
      procedure create_data_store_gen1, create_data_store_gen2
  end interface
contains
  function create_data_store_gen1(arg0, arg1, arg2, arg3) result(data_store)
    use iso_c_binding
    use array_descriptor
    integer(c_int), value :: arg0
    integer(c_int), value :: arg1
    integer(c_int), value :: arg2
    real(c_double), dimension(:,:,:), target :: arg3
    type(c_ptr) :: data_store

    data_store = create_data_store_gen_impl1(arg0, arg1, arg2, create_array_descriptor(arg3))
  end function
  function create_data_store_gen2(arg0, arg1, arg2, arg3) result(data_store)
    use iso_c_binding
    use array_descriptor
    integer(c_int), value :: arg0
    integer(c_int), value :: arg1
    integer(c_int), value :: arg2
    real(c_float), dimension(:,:,:), target :: arg3
    type(c_ptr) :: data_store

    data_store = create_data_store_gen_impl2(arg0, arg1, arg2, create_array_descriptor(arg3))
  end function
end

program main
    use iso_c_binding
    use descriptor_no_generic
    use descriptor_generic
    implicit none

    integer, parameter :: i = 9, j = 10, k = 11
    real(c_double), dimension(i, j, k), target :: in

    type(c_ptr) :: descriptor
    descriptor = create_data_store(i, j, k, in)
    descriptor = create_data_store_gen(i, j, k, in)
end program
