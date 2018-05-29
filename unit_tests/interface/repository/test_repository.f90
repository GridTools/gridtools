function call_repository() bind (c, name="call_repository") &
        result(ret)
    use iso_c_binding
    use repository

    type(c_ptr) :: repository_handle
    real(c_double), dimension(3, 4, 5) :: ijkfield
    real(c_double), dimension(3, 4) :: ijfield
    real(c_double), dimension(4, 5) :: jkfield

    logical(c_bool) :: ret

    ijkfield = reshape( (/ (I, I = 0, 3*4*5) /), shape(ijkfield), (/ 0 /) )
    ijfield = reshape( (/ (I, I = 0, 3*4) /), shape(ijfield), (/ 0 /) )
    jkfield = reshape( (/ (I, I = 0, 4*5) /), shape(jkfield), (/ 0 /) )

    repository_handle = make_exported_repository(3, 4, 5)
    call set_exported_repository_ijkfield(repository_handle, ijkfield)
    call set_exported_repository_ijfield(repository_handle, ijfield)
    call set_exported_repository_jkfield(repository_handle, jkfield)
    ret = verify_exported_repository(repository_handle)

end function
