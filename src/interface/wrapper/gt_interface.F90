
module gt_interface
implicit none
  interface

    subroutine gt_pull_bool(arg0, arg1, arg2, arg3, arg4, arg5) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      CHARACTER(KIND=C_CHAR), dimension(*) :: arg1
      logical(c_bool), dimension(*) :: arg2
      integer(c_int), value :: arg3
      integer(c_int), dimension(*) :: arg4
      integer(c_int), dimension(*) :: arg5
    end
    subroutine gt_pull_double(arg0, arg1, arg2, arg3, arg4, arg5) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      CHARACTER(KIND=C_CHAR), dimension(*) :: arg1
      real(c_double), dimension(*) :: arg2
      integer(c_int), value :: arg3
      integer(c_int), dimension(*) :: arg4
      integer(c_int), dimension(*) :: arg5
    end
    subroutine gt_pull_float(arg0, arg1, arg2, arg3, arg4, arg5) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      CHARACTER(KIND=C_CHAR), dimension(*) :: arg1
      real(c_float), dimension(*) :: arg2
      integer(c_int), value :: arg3
      integer(c_int), dimension(*) :: arg4
      integer(c_int), dimension(*) :: arg5
    end
    subroutine gt_pull_int(arg0, arg1, arg2, arg3, arg4, arg5) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      CHARACTER(KIND=C_CHAR), dimension(*) :: arg1
      integer(c_int), dimension(*) :: arg2
      integer(c_int), value :: arg3
      integer(c_int), dimension(*) :: arg4
      integer(c_int), dimension(*) :: arg5
    end
    subroutine gt_push_bool(arg0, arg1, arg2, arg3, arg4, arg5, arg6) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      CHARACTER(KIND=C_CHAR), dimension(*) :: arg1
      logical(c_bool), dimension(*) :: arg2
      integer(c_int), value :: arg3
      integer(c_int), dimension(*) :: arg4
      integer(c_int), dimension(*) :: arg5
      logical(c_bool), value :: arg6
    end
    subroutine gt_push_double(arg0, arg1, arg2, arg3, arg4, arg5, arg6) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      CHARACTER(KIND=C_CHAR), dimension(*) :: arg1
      real(c_double), dimension(*) :: arg2
      integer(c_int), value :: arg3
      integer(c_int), dimension(*) :: arg4
      integer(c_int), dimension(*) :: arg5
      logical(c_bool), value :: arg6
    end
    subroutine gt_push_float(arg0, arg1, arg2, arg3, arg4, arg5, arg6) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      CHARACTER(KIND=C_CHAR), dimension(*) :: arg1
      real(c_float), dimension(*) :: arg2
      integer(c_int), value :: arg3
      integer(c_int), dimension(*) :: arg4
      integer(c_int), dimension(*) :: arg5
      logical(c_bool), value :: arg6
    end
    subroutine gt_push_int(arg0, arg1, arg2, arg3, arg4, arg5, arg6) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
      CHARACTER(KIND=C_CHAR), dimension(*) :: arg1
      integer(c_int), dimension(*) :: arg2
      integer(c_int), value :: arg3
      integer(c_int), dimension(*) :: arg4
      integer(c_int), dimension(*) :: arg5
      logical(c_bool), value :: arg6
    end
    subroutine gt_run(arg0) bind(c)
      use iso_c_binding
      type(c_ptr), value :: arg0
    end

  end interface
end
