function(gridtools_add_storage_test tgt_name)
    set(options)
    set(one_value_args)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    foreach(storage IN LISTS GT_STORAGES)
        set(tgt ${tgt_name}_${storage})
        gridtools_add_unit_test(${tgt}
                SOURCES ${ARGS_SOURCES}
                LIBRARIES storage_${storage}
                LABELS storage ${storage}
                NO_NVCC)
        string(TOUPPER ${storage} u_storage)
        target_compile_definitions(${tgt} PRIVATE GT_STORAGE_${u_storage})
    endforeach()
endfunction()

gridtools_add_unit_test(test_storage_info SOURCES test_storage_info.cpp LABELS storage)

gridtools_add_storage_test(test_storage_sid SOURCES test_storage_sid.cpp)
gridtools_add_storage_test(test_storage_facility SOURCES test_storage_facility.cpp)
gridtools_add_storage_test(test_alignment_inner_region SOURCES test_alignment_inner_region.cpp)
gridtools_add_storage_test(test_data_store SOURCES test_data_store.cpp)
gridtools_add_storage_test(test_host_view SOURCES test_host_view.cpp)

if(TARGET storage_cuda)
    gridtools_add_storage_test(test_target_view SOURCES test_target_view.cu LIBRARIES storage_cuda LABELS storage cuda)
endif()