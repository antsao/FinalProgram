all:
	g++ Fluid.cpp GLSL_helper.cpp -DGL_GLEXT_PROTOTYPES -framework OpenGL -framework GLUT -Wno-deprecated -Wno-parentheses
linux:
	g++ FLuid.cpp GLSL_helper.cpp -DGL_GLEXT_PROTOTYPES -lGL -lGLU -lglut
