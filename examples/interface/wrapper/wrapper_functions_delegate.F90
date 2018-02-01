! This is needed for now as the c_bindings do not allow to export a polymorphic function
! and do not support to autmoatically deduce the array shape between C and Fortran

MODULE gt_interface_delegate

USE iso_c_binding
USE gt_interface

IMPLICIT NONE

PUBLIC :: &
  gt_push, &
  gt_pull
  
!PRIVATE

INTERFACE gt_push
    MODULE PROCEDURE &
      gt_push_float_3d
!      gt_push_float_1d, &
!      gt_push_float_2d, &
!      gt_push_double_1d, &
!      gt_push_double_2d, &
!      gt_push_double_3d, &
!      gt_push_int_1d, &
!      gt_push_int_2d, &
!      gt_push_int_3d, &
!      gt_push_bool_1d, &
!      gt_push_bool_2d, &
!      gt_push_bool_3d
END INTERFACE



INTERFACE gt_pull
    MODULE PROCEDURE &
      gt_pull_float_3d
!      gt_pull_float_1d, &
!      gt_pull_float_2d, &
!      gt_pull_double_1d, &
!      gt_pull_double_2d, &
!      gt_pull_double_3d, &
!      gt_pull_int_1d, &
!      gt_pull_int_2d, &
!      gt_pull_int_3d, &
!      gt_pull_bool_1d, &
!      gt_pull_bool_2d, &
!      gt_pull_bool_3d
END INTERFACE

CONTAINS

SUBROUTINE gt_push_float_3d(handle,name,data)
  USE iso_c_binding
  
  TYPE(c_ptr), value :: handle
  CHARACTER(LEN=*), INTENT(IN)           :: name
  REAL(KIND=C_FLOAT), INTENT(IN), TARGET :: data(:,:,:)
  
  !local
  INTEGER, DIMENSION(3), TARGET          :: shape_
  INTEGER, DIMENSION(3), TARGET          :: strides_
  logical(c_bool)                 :: force_copy_
  
  force_copy_ = .TRUE.
  shape_ = SHAPE(data)
  
  !TODO proper computation
  strides_(1) = 1
  strides_(2) = shape_(1)
  strides_(3) = shape_(2)*strides_(2)
  
  !TODO fix strides
  CALL gt_push_float( handle, TRIM(name)//C_NULL_CHAR, data, SIZE(shape_), shape_, strides_, force_copy_ )
  
END SUBROUTINE gt_push_float_3d

SUBROUTINE gt_pull_float_3d(handle,name,data)
  USE iso_c_binding
  
  TYPE(c_ptr), value :: handle
  CHARACTER(LEN=*), INTENT(IN)           :: name
  REAL(KIND=C_FLOAT), INTENT(IN), TARGET :: data(:,:,:)
  
  !local
  INTEGER, DIMENSION(3), TARGET          :: shape_
  INTEGER, DIMENSION(3), TARGET          :: strides_
  
  shape_ = SHAPE(data)
  
  !TODO proper computation
  strides_(1) = 1
  strides_(2) = shape_(1)
  strides_(3) = shape_(2)*strides_(2)
  
  !TODO fix strides
  CALL gt_pull_float( handle, TRIM(name)//C_NULL_CHAR, data, SIZE(shape_), shape_, strides_ )
  
END SUBROUTINE gt_pull_float_3d

SUBROUTINE gt_push_double_3d(handle,name,data)
  USE iso_c_binding
  
  TYPE(c_ptr), value :: handle
  CHARACTER(LEN=*), INTENT(IN)           :: name
  REAL(KIND=C_DOUBLE), INTENT(IN), TARGET :: data(:,:,:)
  
  !local
  INTEGER, DIMENSION(3), TARGET          :: shape_
  INTEGER, DIMENSION(3), TARGET          :: strides_
  logical(c_bool)                 :: force_copy_
  
  force_copy_ = .TRUE.
  shape_ = SHAPE(data)
  
  !TODO proper computation
  strides_(1) = 1
  strides_(2) = shape_(1)
  strides_(3) = shape_(2)*strides_(2)
  
  !TODO fix strides
  CALL gt_push_double( handle, TRIM(name)//C_NULL_CHAR, data, SIZE(shape_), shape_, strides_, force_copy_ )
  
END SUBROUTINE gt_push_double_3d

SUBROUTINE gt_pull_double_3d(handle,name,data)
  USE iso_c_binding
  
  TYPE(c_ptr), value :: handle
  CHARACTER(LEN=*), INTENT(IN)           :: name
  REAL(KIND=C_DOUBLE), INTENT(IN), TARGET :: data(:,:,:)
  
  !local
  INTEGER, DIMENSION(3), TARGET          :: shape_
  INTEGER, DIMENSION(3), TARGET          :: strides_
  
  shape_ = SHAPE(data)
  
  !TODO proper computation
  strides_(1) = 1
  strides_(2) = shape_(1)
  strides_(3) = shape_(2)*strides_(2)
  
  !TODO fix strides
  CALL gt_pull_double( handle, TRIM(name)//C_NULL_CHAR, data, SIZE(shape_), shape_, strides_ )
  
END SUBROUTINE gt_pull_double_3d

SUBROUTINE gt_push_int_3d(handle,name,data)
  USE iso_c_binding
  
  TYPE(c_ptr), value :: handle
  CHARACTER(LEN=*), INTENT(IN)           :: name
  REAL(KIND=C_INT), INTENT(IN), TARGET :: data(:,:,:)
  
  !local
  INTEGER, DIMENSION(3), TARGET          :: shape_
  INTEGER, DIMENSION(3), TARGET          :: strides_
  logical(c_bool)                 :: force_copy_
  
  force_copy_ = .TRUE.
  shape_ = SHAPE(data)
  
  !TODO proper computation
  strides_(1) = 1
  strides_(2) = shape_(1)
  strides_(3) = shape_(2)*strides_(2)
  
  !TODO fix strides
  CALL gt_push_int( handle, TRIM(name)//C_NULL_CHAR, data, SIZE(shape_), shape_, strides_, force_copy_ )
  
END SUBROUTINE gt_push_int_3d

SUBROUTINE gt_pull_int_3d(handle,name,data)
  USE iso_c_binding
  
  TYPE(c_ptr), value :: handle
  CHARACTER(LEN=*), INTENT(IN)           :: name
  REAL(KIND=C_INT), INTENT(IN), TARGET :: data(:,:,:)
  
  !local
  INTEGER, DIMENSION(3), TARGET          :: shape_
  INTEGER, DIMENSION(3), TARGET          :: strides_
  
  shape_ = SHAPE(data)
  
  !TODO proper computation
  strides_(1) = 1
  strides_(2) = shape_(1)
  strides_(3) = shape_(2)*strides_(2)
  
  !TODO fix strides
  CALL gt_pull_int( handle, TRIM(name)//C_NULL_CHAR, data, SIZE(shape_), shape_, strides_ )
  
END SUBROUTINE gt_pull_int_3d



END MODULE gt_interface_delegate

