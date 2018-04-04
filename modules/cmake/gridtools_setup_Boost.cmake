macro( gridtools_setup_boost )

  find_package( Boost 1.58 REQUIRED )

  foreach( dir ${Boost_INCLUDE_DIRS} )
    list( APPEND GRIDTOOLS_SYSTEM_INCLUDE_DIRS "${dir}" )
  endforeach()

endmacro()
