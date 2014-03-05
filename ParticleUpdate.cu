#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <map>
#include "ParticleUpdate.h"
#include "Fluid.h"

#define BOTTOM_BOUND -30
#define TOP_BOUND 30
#define FRONT_BOUND 30
#define BACK_BOUND -30
#define LEFT_BOUND -30
#define RIGHT_BOUND 30

#define NUM_CELLS 226981
#define NUM_CELLS_X4 907924

using namespace std;

void updateParticles(Particle *particles, int size, my_vec3 localExtForce) {
  int *gridCounter = (int *)calloc(sizeof(int), NUM_CELLS);
  int *gridCells = (int *)calloc(sizeof(int), NUM_CELLS_X4);

  updateGrid(particles, gridCounter, gridCells);

  int *d_gridCounter;
  if (cudaMalloc(&d_gridCounter, sizeof(int) * NUM_CELLS) != cudaSuccess) {
    printf("didn't malloc space for grid counters\n");
  }
  if (cudaMemcpy(d_gridCounter, gridCounter, 
                 sizeof(int) * NUM_CELLS,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("didn't copy gridCounter\n");
  }

  int *d_gridCells;
  if (cudaMalloc(&d_gridCells, sizeof(int) * NUM_CELLS_X4) != cudaSuccess) {
    printf("didn't malloc space for grid cells\n");
  }
  if (cudaMemcpy(d_gridCells, gridCells, 
                 sizeof(int) * NUM_CELLS_X4,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("didn't copy grid cells");
  }

  Particle *d_particles;
  if (cudaMalloc(&d_particles, sizeof(Particle) * size) != cudaSuccess) {
    printf("didn't malloc space for device particles\n");
  }
  if (cudaMemcpy(d_particles, particles, 
                 sizeof(Particle) * size, 
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("didn't copy particles\n");
  }

  my_vec3 *d_localExtForce;
  if (cudaMalloc(&d_localExtForce, sizeof(my_vec3)) != cudaSuccess) {
    printf("didn't malloc space for force\n");
  }
  if (cudaMemcpy(d_localExtForce, &localExtForce, 
                 sizeof(my_vec3), 
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("didn't copy force\n");
  }
  dim3 dimBlockCells(61);
  dim3 dimGridCells(3721);

  updateParticleInCells<<<dimGridCells, dimBlockCells>>>(d_particles, d_gridCounter, d_gridCells);

  dim3 dimBlock(1024);
  dim3 dimGrid(64);

  // Updating the particles with gravity
  updateParticleKernel<<<dimGrid, dimBlock>>>(d_particles, d_localExtForce);
  cudaMemcpy(particles, d_particles, sizeof(Particle) * size, cudaMemcpyDeviceToHost);

  free(gridCounter);
  free(gridCells);
  cudaFree(d_gridCounter);
  cudaFree(d_gridCells);
  cudaFree(d_particles);
  cudaFree(d_localExtForce);
}

void updateGrid(Particle *particles, int *gridCounter, int *gridCells) {
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

__global__ void updateParticleInCells(Particle *particles, int *gridCounter, int *gridCells) {
  int partArr[4];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int max = gridCounter[idx];
  float velocityArr[12];
  float velocityX = 0;
  float velocityY = 0;
  float velocityZ = 0;

  for (int x = 0; x < max; x++) {
    partArr[x] = gridCells[idx * 4 + x];
  }

  for (int x = 0; x < max; x++) {
    for (int z = 0; z < max; z++) {
      if (x != z) {
        velocityX += particles[partArr[z]].velocity.x;
        velocityY += particles[partArr[z]].velocity.y;
        velocityZ += particles[partArr[z]].velocity.z;
      }
    }
    if (max > 1) {
      velocityArr[x * 3] = velocityX;
      velocityArr[x * 3 + 1] = velocityY;
      velocityArr[x * 3 + 2] = velocityZ;
      velocityX = 0;
      velocityY = 0;
      velocityZ = 0;
    }
  }

  for (int x = 0; x < max; x++) {
    if(max > 1) {
      particles[partArr[x]].velocity.x = velocityArr[x * 3];
      particles[partArr[x]].velocity.y = velocityArr[x * 3 + 1];
      particles[partArr[x]].velocity.z = velocityArr[x * 3 + 2];
    }
  }
}

__global__ void updateParticleKernel(Particle *particles, my_vec3 *extForce) {
  float deltaTime = 0.1;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  Particle part = particles[idx];

  part.velocity.x = part.velocity.x + extForce->x * deltaTime;
  part.velocity.y = part.velocity.y + extForce->y * deltaTime;
  part.velocity.z = part.velocity.z + extForce->z * deltaTime;

  part.position.x = part.position.x + part.velocity.x * deltaTime;
  part.position.y = part.position.y + part.velocity.y * deltaTime;
  part.position.z = part.position.z + part.velocity.z * deltaTime;

  if (part.position.y <= BOTTOM_BOUND) {
    part.position.y = BOTTOM_BOUND;
    part.velocity.y = -part.velocity.y/10000.0;
  }
  if (part.position.y >= TOP_BOUND) {
    part.position.y = TOP_BOUND;
    part.velocity.y = -part.velocity.y/10000.0;
  }
  if (part.position.x <= LEFT_BOUND) {
    part.position.x = LEFT_BOUND;
    part.velocity.x = -part.velocity.x/10000.0;
  }
  if (part.position.x >= RIGHT_BOUND) {
    part.position.x = RIGHT_BOUND;
    part.velocity.x = -part.velocity.x/10000.0;
  }
  if (part.position.z <= BACK_BOUND) {
    part.position.z = BACK_BOUND;
    part.velocity.z = -part.velocity.z/10000.0;
  }
  if (part.position.z >= FRONT_BOUND) {
    part.position.z = FRONT_BOUND;
    part.velocity.z = -part.velocity.z/10000.0;
  }
  particles[idx] = part;
}
