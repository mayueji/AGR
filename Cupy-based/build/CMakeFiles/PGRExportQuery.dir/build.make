# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home1/mayj/PGR-NEW/pgrchange/PGRchange

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home1/mayj/PGR-NEW/pgrchange/PGRchange/build

# Include any dependencies generated for this target.
include CMakeFiles/PGRExportQuery.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/PGRExportQuery.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PGRExportQuery.dir/flags.make

CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o: CMakeFiles/PGRExportQuery.dir/flags.make
CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o: ../src/PGRExportQuery.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home1/mayj/PGR-NEW/pgrchange/PGRchange/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o -c /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/PGRExportQuery.cpp

CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/PGRExportQuery.cpp > CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.i

CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/PGRExportQuery.cpp -o CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.s

CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o.requires:

.PHONY : CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o.requires

CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o.provides: CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o.requires
	$(MAKE) -f CMakeFiles/PGRExportQuery.dir/build.make CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o.provides.build
.PHONY : CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o.provides

CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o.provides.build: CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o


CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o: CMakeFiles/PGRExportQuery.dir/flags.make
CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o: ../src/Cube.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home1/mayj/PGR-NEW/pgrchange/PGRchange/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o -c /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Cube.cpp

CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Cube.cpp > CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.i

CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Cube.cpp -o CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.s

CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o.requires:

.PHONY : CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o.requires

CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o.provides: CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o.requires
	$(MAKE) -f CMakeFiles/PGRExportQuery.dir/build.make CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o.provides.build
.PHONY : CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o.provides

CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o.provides.build: CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o


CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o: CMakeFiles/PGRExportQuery.dir/flags.make
CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o: ../src/Geometry.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home1/mayj/PGR-NEW/pgrchange/PGRchange/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o -c /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Geometry.cpp

CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Geometry.cpp > CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.i

CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Geometry.cpp -o CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.s

CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o.requires:

.PHONY : CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o.requires

CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o.provides: CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o.requires
	$(MAKE) -f CMakeFiles/PGRExportQuery.dir/build.make CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o.provides.build
.PHONY : CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o.provides

CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o.provides.build: CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o


CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o: CMakeFiles/PGRExportQuery.dir/flags.make
CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o: ../src/MarchingCubes.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home1/mayj/PGR-NEW/pgrchange/PGRchange/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o -c /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/MarchingCubes.cpp

CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/MarchingCubes.cpp > CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.i

CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/MarchingCubes.cpp -o CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.s

CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o.requires:

.PHONY : CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o.requires

CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o.provides: CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o.requires
	$(MAKE) -f CMakeFiles/PGRExportQuery.dir/build.make CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o.provides.build
.PHONY : CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o.provides

CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o.provides.build: CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o


CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o: CMakeFiles/PGRExportQuery.dir/flags.make
CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o: ../src/Mesh.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home1/mayj/PGR-NEW/pgrchange/PGRchange/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o -c /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Mesh.cpp

CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Mesh.cpp > CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.i

CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Mesh.cpp -o CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.s

CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o.requires:

.PHONY : CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o.requires

CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o.provides: CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o.requires
	$(MAKE) -f CMakeFiles/PGRExportQuery.dir/build.make CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o.provides.build
.PHONY : CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o.provides

CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o.provides.build: CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o


CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o: CMakeFiles/PGRExportQuery.dir/flags.make
CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o: ../src/Octnode.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home1/mayj/PGR-NEW/pgrchange/PGRchange/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o -c /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Octnode.cpp

CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Octnode.cpp > CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.i

CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Octnode.cpp -o CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.s

CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o.requires:

.PHONY : CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o.requires

CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o.provides: CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o.requires
	$(MAKE) -f CMakeFiles/PGRExportQuery.dir/build.make CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o.provides.build
.PHONY : CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o.provides

CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o.provides.build: CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o


CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o: CMakeFiles/PGRExportQuery.dir/flags.make
CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o: ../src/Octree.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home1/mayj/PGR-NEW/pgrchange/PGRchange/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o -c /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Octree.cpp

CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Octree.cpp > CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.i

CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/Octree.cpp -o CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.s

CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o.requires:

.PHONY : CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o.requires

CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o.provides: CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o.requires
	$(MAKE) -f CMakeFiles/PGRExportQuery.dir/build.make CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o.provides.build
.PHONY : CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o.provides

CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o.provides.build: CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o


CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o: CMakeFiles/PGRExportQuery.dir/flags.make
CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o: ../src/ply.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home1/mayj/PGR-NEW/pgrchange/PGRchange/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o -c /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/ply.cpp

CMakeFiles/PGRExportQuery.dir/src/ply.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PGRExportQuery.dir/src/ply.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/ply.cpp > CMakeFiles/PGRExportQuery.dir/src/ply.cpp.i

CMakeFiles/PGRExportQuery.dir/src/ply.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PGRExportQuery.dir/src/ply.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/ply.cpp -o CMakeFiles/PGRExportQuery.dir/src/ply.cpp.s

CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o.requires:

.PHONY : CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o.requires

CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o.provides: CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o.requires
	$(MAKE) -f CMakeFiles/PGRExportQuery.dir/build.make CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o.provides.build
.PHONY : CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o.provides

CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o.provides.build: CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o


CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o: CMakeFiles/PGRExportQuery.dir/flags.make
CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o: ../src/plyfile.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home1/mayj/PGR-NEW/pgrchange/PGRchange/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o -c /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/plyfile.cpp

CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/plyfile.cpp > CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.i

CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/plyfile.cpp -o CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.s

CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o.requires:

.PHONY : CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o.requires

CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o.provides: CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o.requires
	$(MAKE) -f CMakeFiles/PGRExportQuery.dir/build.make CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o.provides.build
.PHONY : CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o.provides

CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o.provides.build: CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o


CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o: CMakeFiles/PGRExportQuery.dir/flags.make
CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o: ../src/cnpy/cnpy.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home1/mayj/PGR-NEW/pgrchange/PGRchange/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o -c /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/cnpy/cnpy.cpp

CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/cnpy/cnpy.cpp > CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.i

CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home1/mayj/PGR-NEW/pgrchange/PGRchange/src/cnpy/cnpy.cpp -o CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.s

CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o.requires:

.PHONY : CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o.requires

CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o.provides: CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o.requires
	$(MAKE) -f CMakeFiles/PGRExportQuery.dir/build.make CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o.provides.build
.PHONY : CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o.provides

CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o.provides.build: CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o


# Object files for target PGRExportQuery
PGRExportQuery_OBJECTS = \
"CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o" \
"CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o" \
"CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o" \
"CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o" \
"CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o" \
"CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o" \
"CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o" \
"CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o" \
"CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o" \
"CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o"

# External object files for target PGRExportQuery
PGRExportQuery_EXTERNAL_OBJECTS =

../apps/PGRExportQuery: CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o
../apps/PGRExportQuery: CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o
../apps/PGRExportQuery: CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o
../apps/PGRExportQuery: CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o
../apps/PGRExportQuery: CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o
../apps/PGRExportQuery: CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o
../apps/PGRExportQuery: CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o
../apps/PGRExportQuery: CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o
../apps/PGRExportQuery: CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o
../apps/PGRExportQuery: CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o
../apps/PGRExportQuery: CMakeFiles/PGRExportQuery.dir/build.make
../apps/PGRExportQuery: /usr/lib/x86_64-linux-gnu/libz.so
../apps/PGRExportQuery: CMakeFiles/PGRExportQuery.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home1/mayj/PGR-NEW/pgrchange/PGRchange/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX executable ../apps/PGRExportQuery"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/PGRExportQuery.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PGRExportQuery.dir/build: ../apps/PGRExportQuery

.PHONY : CMakeFiles/PGRExportQuery.dir/build

CMakeFiles/PGRExportQuery.dir/requires: CMakeFiles/PGRExportQuery.dir/src/PGRExportQuery.cpp.o.requires
CMakeFiles/PGRExportQuery.dir/requires: CMakeFiles/PGRExportQuery.dir/src/Cube.cpp.o.requires
CMakeFiles/PGRExportQuery.dir/requires: CMakeFiles/PGRExportQuery.dir/src/Geometry.cpp.o.requires
CMakeFiles/PGRExportQuery.dir/requires: CMakeFiles/PGRExportQuery.dir/src/MarchingCubes.cpp.o.requires
CMakeFiles/PGRExportQuery.dir/requires: CMakeFiles/PGRExportQuery.dir/src/Mesh.cpp.o.requires
CMakeFiles/PGRExportQuery.dir/requires: CMakeFiles/PGRExportQuery.dir/src/Octnode.cpp.o.requires
CMakeFiles/PGRExportQuery.dir/requires: CMakeFiles/PGRExportQuery.dir/src/Octree.cpp.o.requires
CMakeFiles/PGRExportQuery.dir/requires: CMakeFiles/PGRExportQuery.dir/src/ply.cpp.o.requires
CMakeFiles/PGRExportQuery.dir/requires: CMakeFiles/PGRExportQuery.dir/src/plyfile.cpp.o.requires
CMakeFiles/PGRExportQuery.dir/requires: CMakeFiles/PGRExportQuery.dir/src/cnpy/cnpy.cpp.o.requires

.PHONY : CMakeFiles/PGRExportQuery.dir/requires

CMakeFiles/PGRExportQuery.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/PGRExportQuery.dir/cmake_clean.cmake
.PHONY : CMakeFiles/PGRExportQuery.dir/clean

CMakeFiles/PGRExportQuery.dir/depend:
	cd /home1/mayj/PGR-NEW/pgrchange/PGRchange/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home1/mayj/PGR-NEW/pgrchange/PGRchange /home1/mayj/PGR-NEW/pgrchange/PGRchange /home1/mayj/PGR-NEW/pgrchange/PGRchange/build /home1/mayj/PGR-NEW/pgrchange/PGRchange/build /home1/mayj/PGR-NEW/pgrchange/PGRchange/build/CMakeFiles/PGRExportQuery.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/PGRExportQuery.dir/depend

