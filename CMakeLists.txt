cmake_minimum_required(VERSION 3.7)

# Set global CMake properties
# -----------------------------------------------------------------------
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake targets")
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# -----------------------------------------------------------------------

# Set global options
# -----------------------------------------------------------------------
option(BUILD_GO             "Build runners"                 ON)
# -----------------------------------------------------------------------

# Set compiler flags
# -----------------------------------------------------------------------
if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
endif()
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Runtime output directory
# -----------------------------------------------------------------------
SET(
  CMAKE_RUNTIME_OUTPUT_DIRECTORY
  "${PROJECT_ROOT_DIR}/bin.${CMAKE_BUILD_TYPE}")
SET(
  CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG
  "${PROJECT_ROOT_DIR}/bin.debug")
SET(
  CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE
  "${PROJECT_ROOT_DIR}/bin.release")
SET(
  CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO
  "${PROJECT_ROOT_DIR}/bin.relwithdebinfo")

# -----------------------------------------------------------------------
# Set library output directories
# -----------------------------------------------------------------------
SET(
  CMAKE_LIBRARY_OUTPUT_DIRECTORY
  "${CMAKE_SOURCE_DIR}/bin.${CMAKE_BUILD_TYPE}")
SET(
  CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG
  "${CMAKE_SOURCE_DIR}/bin.debug")
SET(
  CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE
  "${CMAKE_SOURCE_DIR}/bin.release")
SET(
  CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO
  "${CMAKE_SOURCE_DIR}/bin.relwithdebinfo")

# -----------------------------------------------------------------------
# Set archive output directory
# -----------------------------------------------------------------------
SET(
  CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  "${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE}")
SET(
  CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG
  "${CMAKE_SOURCE_DIR}/lib/debug")
SET(
  CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE
  "${CMAKE_SOURCE_DIR}/lib/release")
SET(
  CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO
  "${CMAKE_SOURCE_DIR}/lib/relwithdebinfo")

# -----------------------------------------------------------------------
# Set install directories
# -----------------------------------------------------------------------
SET(CMAKE_INSTALL_PREFIX "${PROJECT_ROOT_DIR}/install.${CMAKE_BUILD_TYPE}")

# Specify include directories
# -----------------------------------------------------------------------
include_directories(
  ${CMAKE_SOURCE_DIR}/core
  ${CMAKE_SOURCE_DIR}/thirdparty
)
# -----------------------------------------------------------------------

# Add module directories
# -----------------------------------------------------------------------
add_subdirectory(core)
add_subdirectory(thirdparty)
# -----------------------------------------------------------------------