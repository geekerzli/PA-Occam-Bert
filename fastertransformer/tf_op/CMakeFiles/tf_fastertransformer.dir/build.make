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
include fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/depend.make

# Include the progress variables for this target.
include fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/progress.make

# Include the compile flags for this target's objects.
include fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/flags.make

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/flags.make
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o: ../fastertransformer/tf_op/bert_transformer_op.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/ix/FasterTransformer-master_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o"
	cd /workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o -c /workspace/ix/FasterTransformer-master_2/fastertransformer/tf_op/bert_transformer_op.cc

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.i"
	cd /workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/ix/FasterTransformer-master_2/fastertransformer/tf_op/bert_transformer_op.cc > CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.i

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.s"
	cd /workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/ix/FasterTransformer-master_2/fastertransformer/tf_op/bert_transformer_op.cc -o CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.s

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o.requires:

.PHONY : fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o.requires

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o.provides: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o.requires
	$(MAKE) -f fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/build.make fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o.provides.build
.PHONY : fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o.provides

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o.provides.build: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o


fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/flags.make
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o: ../fastertransformer/tf_op/bert_transformer_op.cu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/ix/FasterTransformer-master_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o"
	cd /workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o -c /workspace/ix/FasterTransformer-master_2/fastertransformer/tf_op/bert_transformer_op.cu.cc

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.i"
	cd /workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/ix/FasterTransformer-master_2/fastertransformer/tf_op/bert_transformer_op.cu.cc > CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.i

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.s"
	cd /workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/ix/FasterTransformer-master_2/fastertransformer/tf_op/bert_transformer_op.cu.cc -o CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.s

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o.requires:

.PHONY : fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o.requires

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o.provides: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o.requires
	$(MAKE) -f fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/build.make fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o.provides.build
.PHONY : fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o.provides

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o.provides.build: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o


fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/flags.make
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o: ../fastertransformer/cuda/open_attention.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/ix/FasterTransformer-master_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o"
	cd /workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /workspace/ix/FasterTransformer-master_2/fastertransformer/cuda/open_attention.cu -o CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o.requires:

.PHONY : fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o.requires

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o.provides: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o.requires
	$(MAKE) -f fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/build.make fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o.provides.build
.PHONY : fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o.provides

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o.provides.build: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o


fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/flags.make
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o: ../fastertransformer/cuda/cuda_kernels.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/ix/FasterTransformer-master_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o"
	cd /workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /workspace/ix/FasterTransformer-master_2/fastertransformer/cuda/cuda_kernels.cu -o CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o.requires:

.PHONY : fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o.requires

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o.provides: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o.requires
	$(MAKE) -f fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/build.make fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o.provides.build
.PHONY : fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o.provides

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o.provides.build: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o


# Object files for target tf_fastertransformer
tf_fastertransformer_OBJECTS = \
"CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o" \
"CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o" \
"CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o" \
"CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o"

# External object files for target tf_fastertransformer
tf_fastertransformer_EXTERNAL_OBJECTS =

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/cmake_device_link.o: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/cmake_device_link.o: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/cmake_device_link.o: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/cmake_device_link.o: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/cmake_device_link.o: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/build.make
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/cmake_device_link.o: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/ix/FasterTransformer-master_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CUDA device code/workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/cmake_device_link.o"
	cd /workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tf_fastertransformer.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/build: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/cmake_device_link.o

.PHONY : fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/build

# Object files for target tf_fastertransformer
tf_fastertransformer_OBJECTS = \
"CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o" \
"CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o" \
"CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o" \
"CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o"

# External object files for target tf_fastertransformer
tf_fastertransformer_EXTERNAL_OBJECTS =

lib/libtf_fastertransformer.so: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o
lib/libtf_fastertransformer.so: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o
lib/libtf_fastertransformer.so: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o
lib/libtf_fastertransformer.so: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o
lib/libtf_fastertransformer.so: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/build.make
lib/libtf_fastertransformer.so: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/cmake_device_link.o
lib/libtf_fastertransformer.so: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/ix/FasterTransformer-master_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX shared library ../../lib/libtf_fastertransformer.so"
	cd /workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tf_fastertransformer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/build: lib/libtf_fastertransformer.so

.PHONY : fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/build

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/requires: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cc.o.requires
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/requires: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/bert_transformer_op.cu.cc.o.requires
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/requires: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/open_attention.cu.o.requires
fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/requires: fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/__/cuda/cuda_kernels.cu.o.requires

.PHONY : fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/requires

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/clean:
	cd /workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op && $(CMAKE_COMMAND) -P CMakeFiles/tf_fastertransformer.dir/cmake_clean.cmake
.PHONY : fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/clean

fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/depend:
	cd /workspace/ix/FasterTransformer-master_2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/ix/FasterTransformer-master_2 /workspace/ix/FasterTransformer-master_2/fastertransformer/tf_op /workspace/ix/FasterTransformer-master_2/build /workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op /workspace/ix/FasterTransformer-master_2/build/fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : fastertransformer/tf_op/CMakeFiles/tf_fastertransformer.dir/depend

