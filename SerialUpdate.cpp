#include "Fluid.h"
#include "SerialUpdate.h"
#include "glm/glm.hpp"
#include "math.h"

using namespace glm;

void NaiveCollisionDetection(Particle *particles);

void updateParticlesS(Particle *particles, int size, my_vec3 extForce) {
   int *gridCounter = (int *)calloc(sizeof(int), NUM_CELLS);
   int *gridCells = (int *)calloc(sizeof(int), NUM_CELLS_X4);

   //updateGridS(particles, gridCounter, gridCells);
   //updateParticlesInCellsS(particles, gridCounter, gridCells);
   NaiveCollisionDetection(particles);
   updateParticleKernelS(particles, extForce);

   free(gridCounter);
   free(gridCells);
}

bool DetectCollision(Particle a, Particle b) {
   float x, y, z;
   x = a.position.x - b.position.x;
   y = a.position.y - b.position.y;
   z = a.position.z - b.position.z;
   return (x*x + y*y + z*z) < 6;
}

void NaiveCollisionDetection(Particle *particles) {
   for (int i = 0; i < NUM_PARTICLES; i++) {
      particles[i].hit = 0;
   }
   for (int i = 0; i < NUM_PARTICLES; i++) {
      for (int j = 0; j < NUM_PARTICLES; j++) {
         if (i != j) {
            if (particles[i].hit < 5 && DetectCollision(particles[i], particles[j])) {
               particles[i].hit++;
               vec3 v_ab = vec3(vec3(particles[i].velocity.x - particles[j].velocity.x, 
                               particles[i].velocity.y - particles[j].velocity.y, 
                               particles[i].velocity.z - particles[j].velocity.z));

               vec3 col_normal = vec3(particles[i].position.x - particles[j].position.x, 
                               particles[i].position.y - particles[j].position.y, 
                               particles[i].position.z - particles[j].position.z);
               col_normal = normalize(col_normal);
               float force = (-(1+0.05)*dot(v_ab,col_normal))/(dot(col_normal, col_normal)*1.05);
               particles[j].velocity.x = particles[j].velocity.x - force*col_normal.x; 
               particles[j].velocity.y = particles[j].velocity.y - force*col_normal.y; 
               particles[j].velocity.z = particles[j].velocity.z - force*col_normal.z; 
               particles[i].velocity.x = particles[i].velocity.x + force*col_normal.x; 
               particles[i].velocity.y = particles[i].velocity.y + force*col_normal.y; 
               particles[i].velocity.z = particles[i].velocity.z + force*col_normal.z; 
               /*
               float angle = acos((dir.x*particles[j].velocity.x+
                                   dir.y*particles[j].velocity.y+
                                   dir.z*particles[j].velocity.z)/
                                   (sqrt((dir.x+dir.y+dir.z)*
                                   (particles[j].velocity.x+
                                   particles[j].velocity.y+
                                   particles[j].velocity.z))));
               vec3 center = vec3(particles[j].velocity.x*cos(angle), particles[j].velocity.y*cos(angle), particles[j].velocity.z*cos(angle));
               particle[j].velocity.x =  center.x
            */
            }
         }
      }
   }
}
/*
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
*/

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
        particles[idx].velocity.y = -particles[idx].velocity.y/100;
      }
      if (particles[idx].position.y >= TOP_BOUND) {
        particles[idx].position.y = TOP_BOUND;
        particles[idx].velocity.y = -particles[idx].velocity.y/100;
      }
      if (particles[idx].position.x <= LEFT_BOUND) {
        particles[idx].position.x = LEFT_BOUND;
        particles[idx].velocity.x = -particles[idx].velocity.x/100;
      }
      if (particles[idx].position.x >= RIGHT_BOUND) {
        particles[idx].position.x = RIGHT_BOUND;
        particles[idx].velocity.x = -particles[idx].velocity.x/100;
      }
      if (particles[idx].position.z <= BACK_BOUND) {
        particles[idx].position.z = BACK_BOUND;
        particles[idx].velocity.z = -particles[idx].velocity.z/100;
      }
      if (particles[idx].position.z >= FRONT_BOUND) {
        particles[idx].position.z = FRONT_BOUND;
        particles[idx].velocity.z = -particles[idx].velocity.z/100;
      }
   }
}
