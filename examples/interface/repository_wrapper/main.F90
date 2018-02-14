PROGRAM interface_fortran_example

USE gt_import
USE gt_interface
USE gt_interface_delegate
USE iso_c_binding

TYPE(c_ptr) :: dycore
TYPE(c_ptr) :: dycore_repository
TYPE(c_ptr) :: dycore_repository_explicit
INTEGER, TARGET :: dim(3)
REAL*8, DIMENSION(3,4,5) :: in
REAL*8, DIMENSION(3,4,5) :: out
INTEGER :: i,j,k
LOGICAL :: error

dim(1)=3
dim(2)=4
dim(3)=5
print *, "[Fortran] dim: ", dim(1), dim(2), dim(3)
print *, "size=", SIZE(SHAPE(in)), "shape: ", SHAPE(IN)

dycore_repository = alloc_wrapped_dycore_repository(3, dim) ! pass dimensions for the storage
dycore_repository_explicit = convert_dycore_repo(dycore_repository) !this shouldn't be needed in next release
dycore = alloc_mini_dycore(3, dim, dycore_repository_explicit); ! pass dimensions for the grid (which could be different)

DO i = 1 , dim(1)
      DO j = 1, dim(2)
        DO k = 1, dim(3)
          in(i,j,k)=i*100+j*10+k
          out(i,j,k)=0
        ENDDO
      ENDDO
END DO

print *, "in(1,2,3)=", in(1,2,3)

call gt_push(dycore_repository, "in", in)
call gt_push(dycore_repository, "out", out)

call copy_stencil(dycore)

call gt_pull(dycore_repository, "out", out)

call put_a_number(dycore, 3)
call put_a_number(dycore, 8)
call put_a_number(dycore, 1)

call print_numbers(dycore)

call gt_release(dycore_repository)
call gt_release(dycore)

error = .false.
DO i = 1 , dim(1)
      DO j = 1, dim(2)
        DO k = 1, dim(3)
          IF (out(i,j,k) /= in(i,j,k)) THEN
            error = .true.;
            print *, "error in ", i, j, k
          ENDIF
        ENDDO
      ENDDO
END DO
IF( error ) THEN
    print *, "ERROR!"
    stop 1
ELSE
    print *, "verified!"
    stop 0
ENDIF

END PROGRAM interface_fortran_example
