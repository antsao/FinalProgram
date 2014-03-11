#include "SerialUpdate.h"

void updateParticlesS(Particle *particles, int size, my_vec3 extForce) {
   int *gridCounter = (int *)calloc(sizeof(int), NUM_CELLS);
   int *gridCells = (int *)calloc(sizeof(int), NUM_CELLS_X4);

   updateGridS(particles, gridCounter, gridCells);
   updateParticlesInCellsS(particles, gridCounter, gridCells);
   updateParticleKernelS(particles, extForce);

   free(gridCounter);
   free(gridCells);
}

void updateGridS(Particle *particles, int *gridCounter, int *gridCells) {
  unsigned int idx;
  for (int x = 0; x < NUM_PARTICLES; x++) {
    idx = (particles[x].position.z + 30) * 61 * 61 + 
          (particles[x].position.y + 30) * 61 + 
          (particles[x].position.x + 30);
    if (gridCounter[idx] < 4) {
      gridCells[idx * 4 + gridCounter[idx]] = x;
      gridCounter[idx]++;
    }
  }
}

void updateParticlesInCellsS(Particle *particles, int *gridCounter, int *gridCells) {
   int idx, max, partArr[4];
   float velocityArr[12], velocityX, velocityY, velocityZ;

   for (idx = 0; idx < NUM_PARTICLES; idx++) {
      max = gridCounter[idx];
      velocityX = velocityY = velocityZ = 0.0f;
      for (int x = 0; x < max; x++) {
         partArr[x] = gridCells[idx*4 + x];
      }

      for (int x = 0; x < max; x++) {
         for (int z = 0; z < max; z++) {
            if (x != z) {
               velocityX += particles[partArr[z]].velocity.x;
               velocityY += particles[partArr[z]].velocity.y;
               velocityZ += particles[partArr[z]].velocity.z;
            }
         }
         if ( max > 1 ){
            velocityArr[x*3] = velocityX;
            velocityArr[x*3+1] = velocityY;
            velocityArr[x*3+2] = velocityZ;
            velocityX = 0;
            velocityY = 0;
            velocityZ = 0;
         }
      }
   }

   for (int x = 0; x < max; x++) {
      if (max > 1) {
         particles[partArr[x]].velocity.x = velocityArr[x*3];
         particles[partArr[x]].velocity.y = velocityArr[x*3+1];
         particles[partArr[x]].velocity.z = velocityArr[x*3+2];
      }
   }
}


void updateParticleKernelS(Particle *particles, my_vec3 extForce) {
   float deltaTime = 0.1;
   int idx;
   
   for (idx = 0; idx < NUM_PARTICLES; idx++) {
      particles[idx].velocity.x = particles[idx].velocity.x + extForce.x * deltaTime;
      particles[idx].velocity.y = particles[idx].velocity.y + extForce.y * deltaTime;
      particles[idx].velocity.z = particles[idx].velocity.z + extForce.z * deltaTime;
      
      particles[idx].position.x = particles[idx].position.x + particles[idx].velocity.x * deltaTime;
      particles[idx].position.y = particles[idx].position.y + particles[idx].velocity.y * deltaTime;
      particles[idx].position.z = particles[idx].position.z + particles[idx].velocity.z * deltaTime;
  
      if (particles[idx].position.y <= BOTTOM_BOUND) {
        particles[idx].position.y = BOTTOM_BOUND;
        particles[idx].velocity.y = -particles[idx].velocity.y/10000.0;
      }
      if (particles[idx].position.y >= TOP_BOUND) {
        particles[idx].position.y = TOP_BOUND;
        particles[idx].velocity.y = -particles[idx].velocity.y/10000.0;
      }
      if (particles[idx].position.x <= LEFT_BOUND) {
        particles[idx].position.x = LEFT_BOUND;
        particles[idx].velocity.x = -particles[idx].velocity.x/10000.0;
      }
      if (particles[idx].position.x >= RIGHT_BOUND) {
        particles[idx].position.x = RIGHT_BOUND;
        particles[idx].velocity.x = -particles[idx].velocity.x/10000.0;
      }
      if (particles[idx].position.z <= BACK_BOUND) {
        particles[idx].position.z = BACK_BOUND;
        particles[idx].velocity.z = -particles[idx].velocity.z/10000.0;
      }
      if (particles[idx].position.z >= FRONT_BOUND) {
        particles[idx].position.z = FRONT_BOUND;
        particles[idx].velocity.z = -particles[idx].velocity.z/10000.0;
      }
   }
}
