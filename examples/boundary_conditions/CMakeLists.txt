  file(GLOB_RECURSE INC_ALL_ "[a-z|A-Z|0-9]*.hpp")
  list ( APPEND INC_ALL ${INC_ALL_} )
  file(GLOB_RECURSE INC_ALL_ "../../include/[a-z|A-Z|0-9]*.hpp")
  list ( APPEND INC_ALL ${INC_ALL_} )

  list( APPEND SRCFILES boundary-condition_interpolation_FWSCPPDivergence)

  set( GTEST_FILES ../Options.hpp ../Options.cpp)

  add_definitions("-DBENCHMARK")
#  add_definitions("-DGT_CONCURRENT_BC")

  foreach(srcfile IN LISTS SRCFILES)
    add_executable(${srcfile}   ${srcfile}.cpp ${GTEST_FILES} ${INC_ALL})
    target_link_libraries(${srcfile} ${exe_LIBS} gtest_main)
    
    cuda_add_executable(${srcfile}_cuda   ${srcfile}.cu ${GTEST_FILES} ${INC_ALL} OPTIONS ${GPU_SPECIFIC_FLAGS} )
    target_link_libraries(${srcfile}_cuda ${exe_LIBS} gtest_main)
  endforeach(srcfile)