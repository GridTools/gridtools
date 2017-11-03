macro( gridtools_add_option opt )
  set( options )
  set( single_value_args DEFAULT DESCRIPTION )
  set( multi_value_args )
  include( CMakeParseArguments )
  cmake_parse_arguments( _p "${options}" "${single_value_args}" "${multi_value_args}" ${_FIRST_ARG} ${ARGN} )

  if( NOT DEFINED _p_DEFAULT )
    set( _p_DEFAULT ON )
  endif()

  # Check if user explicitly enabled/disabled the option in cache
  get_property( _user_provided_input CACHE ${opt} PROPERTY VALUE SET )

  # In case you want to enforce option to be required
  if( ${opt} MATCHES "REQUIRE" )
    set( ${opt}_REQUIRED 1 CACHE BOOL "" FORCE ) 
    set( ${opt} ON CACHE BOOL "${_p_DESCRIPTION}" FORCE )
  elseif( NOT ${opt} )
    set( ${opt}_REQUIRED 0 CACHE BOOL "" FORCE )
  endif()

  option( ${opt} "${_p_DESCRIPTION}" ${_p_DEFAULT} )

endmacro()