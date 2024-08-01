cmake_minimum_required(VERSION 3.7)
project(atomic_shared_ptr VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-mcx16 -Wall -Wextra -Wpedantic")

find_package(Boost REQUIRED)

add_library(atomic_shared_ptr INTERFACE)
target_include_directories(atomic_shared_ptr INTERFACE
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/atomic_shared_ptrConfigVersion.cmake"
    VERSION 1.0
    COMPATIBILITY AnyNewerVersion
)

install(TARGETS atomic_shared_ptr
	EXPORT atomic_shared_ptrTargets
	LIBRARY DESTINATION lib COMPONENT Runtime
	ARCHIVE DESTINATION lib COMPONENT Development
	RUNTIME DESTINATION bin COMPONENT Runtime
	PUBLIC_HEADER DESTINATION include COMPONENT Development
	BUNDLE DESTINATION bin COMPONENT Runtime
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/atomic_shared_ptrConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/atomic_shared_ptrConfig.cmake"
    INSTALL_DESTINATION lib/cmake/atomic_shared_ptr
)

install(EXPORT atomic_shared_ptrTargets DESTINATION lib/cmake/atomic_shared_ptr)
install(FILES "${PROJECT_BINARY_DIR}/atomic_shared_ptrConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/atomic_shared_ptrConfig.cmake"
        DESTINATION lib/cmake/atomic_shared_ptr)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)


#include_directories(
#	#external/folly
#	#external/AtomicSharedPtr/src
#	.)

add_executable(measure
	#external/folly/folly/PackedSyncPtr.h
	#external/folly/folly/concurrency/AtomicSharedPtr.h
	#external/folly/folly/lang/SafeAssert.h
	#external/folly/folly/lang/SafeAssert.cpp
	#external/folly/folly/lang/ToAscii.h
	#external/folly/folly/lang/ToAscii.cpp
	#external/AtomicSharedPtr/src/atomic_shared_ptr.h
	#external/AtomicSharedPtr/src/fast_logger.h
	test/experiment.h
	test/measure.cpp)
target_link_libraries(measure atomic_shared_ptr)
