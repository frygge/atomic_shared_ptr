cmake_minimum_required(VERSION 3.0)
project(atomic_shared_ptr)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-mcx16 -Wall -Wextra -Wpedantic")

find_package(Boost REQUIRED)

include_directories(
	#external/folly
	#external/AtomicSharedPtr/src
	.)

add_executable(measure
	#external/folly/folly/PackedSyncPtr.h
	#external/folly/folly/concurrency/AtomicSharedPtr.h
	#external/folly/folly/lang/SafeAssert.h
	#external/folly/folly/lang/SafeAssert.cpp
	#external/folly/folly/lang/ToAscii.h
	#external/folly/folly/lang/ToAscii.cpp
	#external/AtomicSharedPtr/src/atomic_shared_ptr.h
	#external/AtomicSharedPtr/src/fast_logger.h
        shared_ptr.h
	test/experiment.h
        test/measure.cpp)

