cmake_minimum_required(VERSION 3.16)
project(ex3_concurrent_counters LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Put multi-config (VS) outputs next to single-config (Ninja/Make) layout when possible.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

option(EX3_TSAN "Build correctness_test with ThreadSanitizer" OFF)

find_package(Threads REQUIRED)

add_library(ex3_counters
  src/factory.cpp
  src/host_info.cpp
)
target_include_directories(ex3_counters PUBLIC src)
target_link_libraries(ex3_counters PUBLIC Threads::Threads)

if(MSVC)
  target_compile_options(ex3_counters PRIVATE /W4 /EHsc /permissive-)
  # Avoid min/max macros interfering with std::.
  target_compile_definitions(ex3_counters PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(ex3_counters PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark PRIVATE ex3_counters)
if(MSVC)
  target_compile_options(benchmark PRIVATE /W4 /EHsc /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(benchmark PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(correctness_test src/correctness_test.cpp)
target_link_libraries(correctness_test PRIVATE ex3_counters)
if(MSVC)
  target_compile_options(correctness_test PRIVATE /W4 /EHsc /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(correctness_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(EX3_TSAN AND NOT MSVC)
  target_compile_options(ex3_counters PRIVATE -fsanitize=thread -g -O1)
  target_link_options(ex3_counters PUBLIC -fsanitize=thread)
  target_compile_options(correctness_test PRIVATE -fsanitize=thread -g -O1)
  target_link_options(correctness_test PRIVATE -fsanitize=thread)
endif()

install(TARGETS benchmark correctness_test RUNTIME DESTINATION bin)
