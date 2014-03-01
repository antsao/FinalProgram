CC=g++
CPPSOURCES=Fluid.cpp GLSL_helper.cpp GeometryCreator.cpp MStackHelp.cpp
CPPFLAGS= -DGL_GLEXT_PROTOTYPES 
APPLE_CPPFLAGS= -framework OpenGL -framework GLUT -Wno-deprecated -Wno-parentheses
LINUX_CPPFLAGS= -lGL -lGLU -lglut

NCC=nvcc
CUSOURCES=ParticleUpdate.cu

all: mac

mac:
	$(NCC) $(CUSOURCES) $(CPPSOURCES) $(CPPFLAGS) $(APPLE_CPPFLAGS) 
	#g++ Fluid.cpp GLSL_helper.cpp GeometryCreator.cpp MStackHelp.cpp -DGL_GLEXT_PROTOTYPES -framework OpenGL -framework GLUT -Wno-deprecated -Wno-parentheses

linux:
	$(NCC) $(CUSOURCES) $(CPPSOURCES) $(CPPFLAGS) $(LINUX_CPPFLAGS)
