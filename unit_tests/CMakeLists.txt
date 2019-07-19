# collect tests
fetch_x86_tests(. LABELS unittest_x86)
fetch_gpu_tests(. LABELS unittest_cuda)

# include all subdirs
if ( COMPONENT_COMMON )
   add_subdirectory( common )
endif()
if ( COMPONENT_GCL )
   add_subdirectory( communication )
endif()
if ( COMPONENT_BOUNDARY_CONDITIONS )
   add_subdirectory( boundary_conditions )
endif()

if ( COMPONENT_DISTRIBUTED_BOUNDARIES )
   add_subdirectory( distributed_boundaries )
endif()
if ( COMPONENT_STENCIL_COMPOSITION )
   add_subdirectory( stencil_composition )
endif()
if ( COMPONENT_STORAGE )
   add_subdirectory( storage )
endif()

if ( COMPONENT_C_BINDINGS )
   add_subdirectory( interface )
endif()
