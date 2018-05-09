program main
    use array_descriptor
    implicit none

    integer, parameter :: i = 9, j = 10, k = 11
    real(8), dimension(i, j, k), target :: in

    type(gt_fortran_array_descriptor) :: descriptor
    descriptor = create_array_descriptor(in)
end program
