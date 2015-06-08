# set(CMAKE_C_COMPILER "llvm-gcc")
# set(CMAKE_CXX_COMPILER "llvm-g++")
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

project(isct_propseg)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE RELEASE CACHE STRING "RELEASE" FORCE)
set(CMAKE_INCLUDE_PATH ${ITK_DIR}/include/ITK-4.5)
set(CMAKE_INCLUDE_PATH ${VTK_DIR}/include/vtk-6.1)
set(CMAKE_INCLUDE_PATH ${ITK_DIR}/include/ITK-4.7)

option(BUILD_SHARED_LIBS OFF)

set(PLATFORM_CHECK true)
if(PLATFORM_CHECK)
  if(DARWIN_MAJOR_VERSION LESS "9")
    message(FATAL_ERROR "Only Mac OSX >= 10.5 are supported !")
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  if (NOT "${CMAKE_CXX_FLAGS}" MATCHES "-fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  endif()
  if(NOT "${CMAKE_C_FLAGS}" MATCHES "-fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()

# set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk")

set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

# This line was set up for statically linking standard libraries with the binary - does not work with clang on OSX10.10
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
set(CMAKE_EXE_LINK_DYNAMIC_C_FLAGS)
set(CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS)
set(CMAKE_SHARED_LIBRARY_C_FLAGS)
set(CMAKE_SHARED_LIBRARY_CXX_FLAGS)
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)


FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

include_directories ("${PROJECT_SOURCE_DIR}/util")

aux_source_directory(. SRC_LIST)
aux_source_directory(./util SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

TARGET_LINK_LIBRARIES(isct_propseg
ITKIONIFTI
ITKIOImageBase
ITKCommon
ITKIONRRD
ITKIOMRC
ITKIOMeta
ITKIOPNG
ITKIOJPEG
ITKIOIPL
ITKIOLSM
ITKIOMesh
ITKIOTIFF
ITKIOSiemens
ITKIOSpatialObjects
ITKIOVTK
ITKIOXML
ITKIOStimulate
ITKIOBioRad
ITKIOBMP
ITKIOCSV
ITKIOGDCM
ITKIOGE
ITKIOGIPL
ITKIOHDF5
ITKOptimizers
vtkalglib
vtkCommonCore
vtkCommonDataModel
vtkIOLegacy
vtkCommonExecutionModel
vtkFiltersCore
vtkFiltersModeling
vtkIOGeometry
vtkFiltersGeneral
)