function( serialization_example target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} SYSTEM PUBLIC ${SERIALBOX_INCLUDE_DIRS})
  target_link_libraries(${target} gtest gtest_main ${SERIALBOX_CXX_LIBRARIES})
endfunction()

serialization_example(serialization_simple serialization_simple.cpp)
serialization_example(serialization_temporaries serialization_temporaries.cpp)
serialization_example(serialization_multistage serialization_multistage.cpp)

serialization_example(serialization_all serialization_simple.cpp 
                                        serialization_temporaries.cpp
                                        serialization_multistage.cpp)

