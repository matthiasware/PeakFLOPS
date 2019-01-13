cmake_minimum_required(VERSION 3.11)
project(PeakFLOPS CXX)

# set(CMAKE_CXX_COMPILER /usr/bin/g++)
# set(CMAKE_CXX_COMPILER /usr/bin/g++)
# set(CMAKE_CXX_COMPILER /usr/bin/clang++)

add_executable(peak_flops peak_flops.cc)
add_executable(test_flops test_flops.cc)

target_compile_features(peak_flops PRIVATE cxx_std_11)
target_compile_features(test_flops PRIVATE cxx_std_11)

message("USING ${CMAKE_CXX_COMPILER_ID}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(FLAGS -Ofast -Wall)
if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
	set(FLAGS ${FLAGS} -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
	set(FLAGS ${FLAGS} -march=core-avx2)
endif()

message(${FLAGS})
target_compile_options(peak_flops PRIVATE ${FLAGS})
target_compile_options(test_flops PRIVATE ${FLAGS})
