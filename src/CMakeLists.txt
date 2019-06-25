add_library(gcl
    GCL.cpp
    )
target_link_libraries(gcl PUBLIC gridtools)
if(GT_USE_MPI)
    target_link_libraries(gcl PUBLIC MPI::MPI_CXX)
endif()

add_subdirectory(tools)
