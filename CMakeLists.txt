cmake_minimum_required(VERSION 3.5)
project(nemoann)

set(CMAKE_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(USE_AIO "Use AIO instead of liburing" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_C_COMPILER_LAUNCHER ccache)
# set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

include_directories(
	${PROJECT_SOURCE_DIR}/include 
	${PROJECT_SOURCE_DIR}/include/tsl/include 
)

add_compile_options(
	-fopenmp 
	-fopenmp-simd
	-Wall
	-Wextra
	-Wfatal-errors
	-Wno-class-memaccess
	-Wno-unused-variable
	-Wno-unused-parameter
	-Wno-unused-but-set-variable
)
add_compile_options(-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -Wl,--no-as-needed)

message("Additional definitions: " $ENV{ADDITIONAL_DEFINITIONS})
add_definitions(
	$ENV{ADDITIONAL_DEFINITIONS}
	-DNDEBUG
	# -DIN_PLACE_RECORD_UPDATE # disable design #1
	-DBG_IO_THREAD # enable opt #1.2

	# -DNO_POLLUTE_ORIGINAL

	# optimizations.
	# -DREAD_ONLY_TESTS
	# -DNO_MAPPING # starling: must disable this!
	-DOVERLAP_INIT
)

include_directories(
	${PROJECT_SOURCE_DIR}/third_party/liburing/src/include
)
link_directories(
	${PROJECT_SOURCE_DIR}/third_party/liburing/src
)

if(USE_AIO)
	message(STATUS "Using AIO for asynchronous I/O, compatible but slower.")
	link_libraries(aio)
	add_definitions(-DUSE_AIO)
else()
	message(STATUS "Using liburing for asynchronous I/O, faster but requires kernel support.")
	link_libraries(uring)
endif()

# Use tcmalloc, as jemalloc does not support python (static TLS block).
link_libraries(
	tcmalloc
)
# add_compile_options(-fsanitize=address,undefined -lasan -lubsan)
# add_link_options(-fsanitize=address,undefined)

# check for AVX512 and AVX2, add the corresponding definitions.
# First, for the compiler.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

# Then, for the runtime.
set(CPU_SUPPORTS_AVX512 FALSE)
set(CPU_SUPPORTS_AVX2 FALSE)
try_run(
	RUN_RESULT
	COMPILE_RESULT
	"${CMAKE_BINARY_DIR}"
	SOURCES "${PROJECT_SOURCE_DIR}/tests/test_cpu.cpp"
	COMPILE_OUTPUT_VARIABLE COMPILE_OUT
	RUN_OUTPUT_VARIABLE RUN_OUT
)

if(COMPILE_RESULT AND RUN_RESULT EQUAL 3) # AVX512 + AVX2
	set(CPU_SUPPORTS_AVX512 TRUE)
	message(STATUS "CPU supports AVX512")
elseif(COMPILE_RESULT AND RUN_RESULT EQUAL 1) # AVX2
	set(CPU_SUPPORTS_AVX2 TRUE)
	message(STATUS "CPU supports AVX2")
else()
	message(STATUS "CPU does not support AVX512 or detection failed")
endif()

if(COMPILER_SUPPORTS_AVX512F AND CPU_SUPPORTS_AVX512)
    message(STATUS "Compiler and CPU supports -mavx512f. Enabling AVX512F.")
	add_definitions(-DUSE_AVX512)
elseif(COMPILER_SUPPORTS_AVX2 AND CPU_SUPPORTS_AVX2)
	message(STATUS "Compiler and CPU supports -mavx2. Enabling AVX2.")
	add_definitions(-DUSE_AVX2)
else()
    message(STATUS "Compiler or CPU do not support -mavx512f and -mavx2.")
endif()

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()

	
find_package(BLAS)
if(BLAS_FOUND)
	message(STATUS "BLAS found ${BLAS_LIBRARIES}")
    if("${BLAS_LIBRARIES}" MATCHES "mkl")
		message(STATUS "BLAS is MKL, adding mkl_rt")
		set(BLAS_LIBRARIES mkl_rt)
    endif()
	link_libraries(${BLAS_LIBRARIES})
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tests/utils)
