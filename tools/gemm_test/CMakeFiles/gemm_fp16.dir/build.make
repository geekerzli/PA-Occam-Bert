# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/ix/FasterTransformer-master_2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/ix/FasterTransformer-master_2/build

# Include any dependencies generated for this target.
include tools/gemm_test/CMakeFiles/gemm_fp16.dir/depend.make

# Include the progress variables for this target.
include tools/gemm_test/CMakeFiles/gemm_fp16.dir/progress.make

# Include the compile flags for this target's objects.
include tools/gemm_test/CMakeFiles/gemm_fp16.dir/flags.make

tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o: tools/gemm_test/CMakeFiles/gemm_fp16.dir/flags.make
tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o: ../tools/gemm_test/gemm_fp16.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/ix/FasterTransformer-master_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o"
	cd /workspace/ix/FasterTransformer-master_2/build/tools/gemm_test && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /workspace/ix/FasterTransformer-master_2/tools/gemm_test/gemm_fp16.cu -o CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o

tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o.requires:

.PHONY : tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o.requires

tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o.provides: tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o.requires
	$(MAKE) -f tools/gemm_test/CMakeFiles/gemm_fp16.dir/build.make tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o.provides.build
.PHONY : tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o.provides

tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o.provides.build: tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o


# Object files for target gemm_fp16
gemm_fp16_OBJECTS = \
"CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o"

# External object files for target gemm_fp16
gemm_fp16_EXTERNAL_OBJECTS =

tools/gemm_test/CMakeFiles/gemm_fp16.dir/cmake_device_link.o: tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o
tools/gemm_test/CMakeFiles/gemm_fp16.dir/cmake_device_link.o: tools/gemm_test/CMakeFiles/gemm_fp16.dir/build.make
tools/gemm_test/CMakeFiles/gemm_fp16.dir/cmake_device_link.o: tools/gemm_test/CMakeFiles/gemm_fp16.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/ix/FasterTransformer-master_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code /workspace/ix/FasterTransformer-master_2/build/tools/gemm_test/CMakeFiles/gemm_fp16.dir/cmake_device_link.o"
	cd /workspace/ix/FasterTransformer-master_2/build/tools/gemm_test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gemm_fp16.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/gemm_test/CMakeFiles/gemm_fp16.dir/build: tools/gemm_test/CMakeFiles/gemm_fp16.dir/cmake_device_link.o

.PHONY : tools/gemm_test/CMakeFiles/gemm_fp16.dir/build

# Object files for target gemm_fp16
gemm_fp16_OBJECTS = \
"CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o"

# External object files for target gemm_fp16
gemm_fp16_EXTERNAL_OBJECTS =

bin/gemm_fp16: tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o
bin/gemm_fp16: tools/gemm_test/CMakeFiles/gemm_fp16.dir/build.make
bin/gemm_fp16: tools/gemm_test/CMakeFiles/gemm_fp16.dir/cmake_device_link.o
bin/gemm_fp16: tools/gemm_test/CMakeFiles/gemm_fp16.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/ix/FasterTransformer-master_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable ../../bin/gemm_fp16"
	cd /workspace/ix/FasterTransformer-master_2/build/tools/gemm_test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gemm_fp16.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/gemm_test/CMakeFiles/gemm_fp16.dir/build: bin/gemm_fp16

.PHONY : tools/gemm_test/CMakeFiles/gemm_fp16.dir/build

tools/gemm_test/CMakeFiles/gemm_fp16.dir/requires: tools/gemm_test/CMakeFiles/gemm_fp16.dir/gemm_fp16.cu.o.requires

.PHONY : tools/gemm_test/CMakeFiles/gemm_fp16.dir/requires

tools/gemm_test/CMakeFiles/gemm_fp16.dir/clean:
	cd /workspace/ix/FasterTransformer-master_2/build/tools/gemm_test && $(CMAKE_COMMAND) -P CMakeFiles/gemm_fp16.dir/cmake_clean.cmake
.PHONY : tools/gemm_test/CMakeFiles/gemm_fp16.dir/clean

tools/gemm_test/CMakeFiles/gemm_fp16.dir/depend:
	cd /workspace/ix/FasterTransformer-master_2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/ix/FasterTransformer-master_2 /workspace/ix/FasterTransformer-master_2/tools/gemm_test /workspace/ix/FasterTransformer-master_2/build /workspace/ix/FasterTransformer-master_2/build/tools/gemm_test /workspace/ix/FasterTransformer-master_2/build/tools/gemm_test/CMakeFiles/gemm_fp16.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/gemm_test/CMakeFiles/gemm_fp16.dir/depend

