add_library(gcl
    GCL.cpp
    )
target_link_libraries(gcl PUBLIC gridtools)
if(GT_USE_MPI)
    target_link_libraries(gcl PUBLIC MPI::MPI_C)
endif()

install(TARGETS gcl EXPORT GridToolsTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

add_subdirectory(tools)
