all:
	g++ Fluid.cpp GLSL_helper.cpp GeometryCreator.cpp MStackHelp.cpp -DGL_GLEXT_PROTOTYPES -framework OpenGL -framework GLUT -Wno-deprecated -Wno-parentheses
linux:
	g++ Fluid.cpp GLSL_helper.cpp GeometryCreator.cpp MStackHelp.cpp -DGL_GLEXT_PROTOTYPES -lGL -lGLU -lglut
