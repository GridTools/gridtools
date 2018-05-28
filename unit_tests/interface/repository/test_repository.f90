function call_repository() bind (c, name="call_repository") &
        result(ret)
    use iso_c_binding
    use repository

    type(c_ptr) :: repository_handle
    real(c_double), dimension(3, 4, 5) :: ijkfield
    real(c_double), dimension(3, 4) :: ijfield

    logical(c_bool) :: ret

    ijkfield = reshape( (/ (I, I = 0, 3*4*5) /), shape(ijkfield), (/ 0 /) )
    ijfield = reshape( (/ (I, I = 0, 3*4) /), shape(ijfield), (/ 0 /) )

    repository_handle = make_exported_repository(3, 4, 5)
    call set_exported_repository_ijkfield(repository_handle, ijkfield)
    call set_exported_repository_ijfield(repository_handle, ijfield)
    ret = verify_exported_repository(repository_handle)

end function
