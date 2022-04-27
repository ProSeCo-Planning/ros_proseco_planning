cmake_minimum_required(VERSION 3.5.0)
project(ros_proseco_planning)

###################################
## OpenMP
###################################
find_package(OpenMP REQUIRED)

###################################
## Include Checks
###################################
include(CheckIncludeFiles)
check_include_files("valgrind/callgrind.h" FOUND_CALLGRIND_H)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/config.h)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    proseco_planning
    roscpp
    roslib
    )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
    CATKIN_DEPENDS proseco_planning
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${catkin_INCLUDE_DIRS}
)

add_executable(proseco_planning_node
    src/main.cpp
    src/prosecoPlanner.cpp
    )

target_compile_features(proseco_planning_node PUBLIC cxx_std_20)
# Specify libraries to link a library or executable target against
target_link_libraries(proseco_planning_node
    ${catkin_LIBRARIES}
    ${proseco_planning_LIBRARIES}
    OpenMP::OpenMP_CXX
)

#############
## INSTALL ##
#############
install(TARGETS proseco_planning_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
install(DIRECTORY python/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/python
  )