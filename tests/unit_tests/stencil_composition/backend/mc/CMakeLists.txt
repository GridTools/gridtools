if(NOT TARGET backend_mc)
    return()
endif()

gridtools_add_unit_test(test_tmp_storage_sid_mc SOURCES test_tmp_storage_sid_mc.cpp LIBRARIES backend_mc)
