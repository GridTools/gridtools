# This is used to generate a pretty target name out of a path
# E.g., /some/folder/test_xy will become some_folder_test_xy
# This is needed because otherwise it can happen that multiple 
# targets with the same names are generated, and this is not legal

function( fix_case_name output case_file )
    get_filename_component( case_dir ${case_file} DIRECTORY )
    get_filename_component( case_name ${case_file} NAME_WE )
    string( LENGTH ${CMAKE_CURRENT_SOURCE_DIR} current_dir_length)
    string( SUBSTRING ${case_dir} ${current_dir_length} -1 case_stripped_dir )
    string( REPLACE "/" "_" case_stripped_dir "${case_stripped_dir}" )
    set(${output} ${case_stripped_dir}${case_name} PARENT_SCOPE )
endfunction( fix_case_name )
                                       
