#include <iostream>

#ifdef __APPLE__
#include "GLUT/glut.h"
#include <OPENGL/gl.h>
#endif

#ifdef __unix__
#include <GL/glut.h>
#endif

#ifdef _WIN32
#pragma comment(lib, "glew32.lib")

#include <GL\glew.h>
#include <GL\glut.h>
#endif

#include <time.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "GLSL_helper.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "GeometryCreator.h"
#include "MStackHelp.h"
#include "Fluid.h"
#include "ParticleUpdate.h"

using namespace std;
using namespace glm;

#define PI 3.14

GLuint triBuffObj;
GLuint colBuffObj;
GLuint normalBuffObj;

float particleYPos = 25;
float particleXPos = -25;
float particleZPos = 25;
int particleCount = 1;
int mode = 2;
int shade = 1;
int ShadeProg;
int mouseStartX;
int mouseStartY;
int mouseEndX;
int mouseEndY;
float g_width;
float g_height;
vec4 directionLight = vec4(0, 0, 0, 0);

ParticleSystem allParticles(-1, NULL);
Mesh *particle;

GLint h_aPosition;
GLint h_aNormal;
GLint h_uModelMatrix;
GLint h_uViewMatrix;
GLint h_uProjMatrix;
GLint h_lightPos;
GLint h_cameraPos;
GLint h_uMatAmb;
GLint h_uMatDif;
GLint h_uMatSpec;
GLint h_uMatShine;
GLint h_uColor;

void SetProjectionMatrix() {
  mat4 Projection = perspective(90.0f, (float)g_width/g_height, 0.1f, 100.f);
  safe_glUniformMatrix4fv(h_uProjMatrix, value_ptr(Projection));
}

void SetView() {
  mat4 Trans = translate(mat4(1.0f), vec3(0.0f, 0, -60));
  mat4 RotateX = rotate(Trans, 0.0f, vec3(0.0f, 1, 0));
  safe_glUniformMatrix4fv(h_uViewMatrix, value_ptr(RotateX));
}

void SetModel(float transX, float transY, float transZ) {
  mat4 trans = translate(mat4(1.0f), vec3(transX, transY, transZ));
  safe_glUniformMatrix4fv(h_uModelMatrix, value_ptr(trans));
}

void SetMaterial() {
  glUseProgram(ShadeProg);
  safe_glUniform3f(h_uMatAmb, 0.2, 0.2, 0.2);
  safe_glUniform3f(h_uMatDif, 0.0, 0.08, 0.5);
  safe_glUniform3f(h_uMatSpec, 0.4, 0.4, 0.4);
  safe_glUniform1f(h_uMatShine, 200.0);
}

int InstallShader(const GLchar *vShaderName, const GLchar *fShaderName) {
  GLuint VS; //handles to shader object
  GLuint FS; //handles to frag shader object
  GLint vCompiled, fCompiled, linked; //status of shader

  VS = glCreateShader(GL_VERTEX_SHADER);
  FS = glCreateShader(GL_FRAGMENT_SHADER);

  //load the source
  glShaderSource(VS, 1, &vShaderName, NULL);
  glShaderSource(FS, 1, &fShaderName, NULL);

  //compile shader and print log
  glCompileShader(VS);
  /* check shader status requires helper functions */
  printOpenGLError();
  glGetShaderiv(VS, GL_COMPILE_STATUS, &vCompiled);
  printShaderInfoLog(VS);

  //compile shader and print log
  glCompileShader(FS);
  /* check shader status requires helper functions */
  printOpenGLError();
  glGetShaderiv(FS, GL_COMPILE_STATUS, &fCompiled);
  printShaderInfoLog(FS);

  if (!vCompiled || !fCompiled) {
    printf("Error compiling either shader %s or %s", vShaderName, fShaderName);
    return 0;
  }

  //create a program object and attach the compiled shader
  ShadeProg = glCreateProgram();
  glAttachShader(ShadeProg, VS);
  glAttachShader(ShadeProg, FS);

  glLinkProgram(ShadeProg);
  /* check shader status requires helper functions */
  printOpenGLError();
  glGetProgramiv(ShadeProg, GL_LINK_STATUS, &linked);
  printProgramInfoLog(ShadeProg);

  glUseProgram(ShadeProg);

  /* get handles to attribute data */
  h_aPosition = safe_glGetAttribLocation(ShadeProg, "aPosition");
  h_aNormal = safe_glGetAttribLocation(ShadeProg, "aNormal");
  h_uProjMatrix = safe_glGetUniformLocation(ShadeProg, "uProjMatrix");
  h_uViewMatrix = safe_glGetUniformLocation(ShadeProg, "uViewMatrix");
  h_uModelMatrix = safe_glGetUniformLocation(ShadeProg, "uModelMatrix");
  h_uColor = safe_glGetUniformLocation(ShadeProg, "uColor");
  /*h_uMatAmb = safe_glGetUniformLocation(ShadeProg, "uMat.aColor");
  h_uMatDif = safe_glGetUniformLocation(ShadeProg, "uMat.dColor");
  h_uMatSpec = safe_glGetUniformLocation(ShadeProg, "uMat.sColor");
  h_uMatShine = safe_glGetUniformLocation(ShadeProg, "uMat.shine");
  h_lightPos = safe_glGetUniformLocation(ShadeProg, "lightPos");*/
  h_cameraPos = safe_glGetUniformLocation(ShadeProg, "cameraPos");
  return 1;
}

void InitGeom() {
  // Make patient ZERO particle
  particle = GeometryCreator::CreateSphere(glm::vec3(0.1f));

  // Fill HouseKeeping Array of Particle positions
  allParticles.initalize();
}

void Initialize() {
  glClearColor(1, 1, 1, 1.0f);
  glEnable(GL_DEPTH_TEST);
}

void Draw() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glUseProgram(ShadeProg);
  SetProjectionMatrix();
  SetView();
  glUniform4f(h_lightPos, directionLight.x, directionLight.y, directionLight.z, 1);
  glUniform4f(h_cameraPos, 0, 0, 0, 1);

  // Draw based on Array of Particle Positions
  for (int index = 0; index < NUM_PARTICLES; index++) {
    safe_glUniform3f(h_uColor, allParticles.particles[index].color.x, 
                               allParticles.particles[index].color.y,
                               allParticles.particles[index].color.z);
    safe_glEnableVertexAttribArray(h_aPosition);
    glBindBuffer(GL_ARRAY_BUFFER, particle->PositionHandle);
    safe_glVertexAttribPointer(h_aPosition, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, particle->IndexHandle);
    SetModel(allParticles.particles[index].position.x, 
             allParticles.particles[index].position.y, 
             allParticles.particles[index].position.z);
    glDrawElements(GL_TRIANGLES, particle->IndexBufferLength, GL_UNSIGNED_SHORT, 0);
  }

  allParticles.update(0.01);

  safe_glDisableVertexAttribArray(h_aPosition);
  glUseProgram(0);
  glutSwapBuffers();
}

void ReshapeGL(int width, int height) {
  g_width = (float)width;
  g_height = (float)height;
  glViewport(0, 0, (GLsizei)(width), (GLsizei)(height));
}

void keyboard(unsigned char key, int x, int y ) {
  switch (key) {
    case 'q': case 'Q' :
      exit( EXIT_SUCCESS );
      break;
  }
  glutPostRedisplay();
}

void update(int val) {
   glutPostRedisplay();
   glutTimerFunc(10, update, 0);
}

int main(int argc, char *argv[]) {
  glutInit(&argc, argv);
  glutInitWindowPosition(200, 200);
  glutInitWindowSize(1000, 1000);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutCreateWindow("Fluid");
  glutReshapeFunc(ReshapeGL);
  glutDisplayFunc(Draw);
  glutKeyboardFunc(keyboard);
  glutTimerFunc(10, update, 0);
  g_width = g_height = 200;
  #ifdef _WIN32
    GLenum err = glewInit();
    if (GLEW_OK != err) {
      cerr << "Error initializing glew! " << glewGetErrorString(err) << endl;
      return 1;
    }
  #endif
  Initialize();
  getGLversion();
  if (!InstallShader(textFileRead((char *)"Fluid_Vert.glsl"), textFileRead((char *)"Fluid_Frag.glsl"))) {
    printf("Error installing shader!\n");
    return 0;
  }
  InitGeom();
  glutMainLoop();
  return 0;
}
