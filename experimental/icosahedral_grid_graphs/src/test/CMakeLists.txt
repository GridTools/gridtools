ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

add_executable(NeighbourListsTest NeighbourListsTest.cpp)
target_link_libraries(NeighbourListsTest ${Boost_LIBRARIES})

