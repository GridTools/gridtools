module gt_handle
    implicit none
    interface
        subroutine gt_release(h) bind(c)
            use iso_c_binding
            type(c_ptr), value :: h
        end
    end interface
end
