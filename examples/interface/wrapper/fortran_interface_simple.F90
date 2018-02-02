PROGRAM interface_fortran_example

#ifdef USE_TYPE_FLOAT
#define DATA_TYPE REAL
#elif USE_TYPE_DOUBLE
#define DATA_TYPE REAL*8
#elif USE_TYPE_INT
#define DATA_TYPE INTEGER
#else
#error "datatype not defined"
#endif

USE gt_import
USE gt_interface
USE gt_interface_delegate
USE iso_c_binding

TYPE(c_ptr) :: dycore_wrapper
INTEGER, TARGET :: dim(3)
DATA_TYPE, DIMENSION(3,4,5) :: in
DATA_TYPE, DIMENSION(3,4,5) :: out
INTEGER :: i,j,k
LOGICAL :: error

dim(1)=3
dim(2)=4
dim(3)=5
print *, "[Fortran] dim: ", dim(1), dim(2), dim(3)
print *, "size=", SIZE(SHAPE(in)), "shape: ", SHAPE(IN)

dycore_wrapper = alloc_simple_wrapper(3, dim)

DO i = 1 , dim(1)
      DO j = 1, dim(2)
        DO k = 1, dim(3)
          in(i,j,k)=i*100+j*10+k
          out(i,j,k)=0
        ENDDO
      ENDDO
END DO

print *, "in(1,2,3)=", in(1,2,3)

call gt_push(dycore_wrapper, "in", in)
call gt_push(dycore_wrapper, "out", out)

call gt_run(dycore_wrapper)

call gt_pull(dycore_wrapper, "out", out)

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
ELSE
    print *, "verified!"
ENDIF

!call gt_wrapper_call_proc(dycore_wrapper, "pass_int", C_LOC(dim))

print *, "[Fortran] dim: ", dim(1), dim(2), dim(3)

!call gt_wrapper_call(dycore_wrapper,"some_action")  
!call gt_wrapper_call(dycore_wrapper,"some_action2")

!call gt_wrapper_call_proc(dycore_wrapper, "pass_int", C_LOC(dim))

END PROGRAM interface_fortran_example
