cmake_minimum_required(VERSION 2.8)

# set the project name (and the name f)
project(Wave)

# add the executable
add_executable(Wave Wave_MPI_timetest.cpp)

find_package(MPI REQUIRED)

target_include_directories(Wave PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(Wave ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(Wave PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(Wave PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()