#include "Fluid.h"

#define BOTTOM_BOUND -30
#define TOP_BOUND 30
#define FRONT_BOUND 30
#define BACK_BOUND -30
#define LEFT_BOUND -30
#define RIGHT_BOUND 30

#define NUM_CELLS 226981
#define NUM_CELLS_X4 907924

void updateParticlesS(Particle *particles, int size, my_vec3 extForce);
void updateGridS(Particle *particles, int *gridCounter, int *gridCells)
void updateParticlesInCellsS(Particle *particles, int *gridCounter, int *gridCells);
void updateParticleKernelS(Particle *particles, my_vec3 extForce);
