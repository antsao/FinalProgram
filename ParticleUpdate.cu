#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "ParticleUpdate.h"
#include "Fluid.h"

#define BOTTOM_BOUND -30
#define FRONT_BOUND 30
#define BACK_BOUND -30
#define LEFT_BOUND -30
#define RIGHT_BOUND 30

void updateParticles(Particle *particles, int size, my_vec3 localExtForce) {
  dim3 dimBlock(256);
  dim3 dimGrid(64);

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

  updateParticleKernel<<<dimGrid, dimBlock>>>(d_particles, d_localExtForce);
  cudaMemcpy(particles, d_particles, sizeof(Particle) * size, cudaMemcpyDeviceToHost);

  cudaFree(d_particles);
  cudaFree(d_localExtForce);
}

__global__ void updateParticleKernel(Particle *particles, my_vec3 *extForce) {
  float deltaTime = 0.01;
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
  }
  if (part.position.x <= LEFT_BOUND) {
    part.position.y = LEFT_BOUND;
  }
  if (part.position.x >= RIGHT_BOUND) {
    part.position.x = RIGHT_BOUND;
  }
  if (part.position.z <= BACK_BOUND) {
    part.position.z = BACK_BOUND;
  }
  if (part.position.z >= FRONT_BOUND) {
    part.position.z = FRONT_BOUND;
  }

  particles[idx] = part;
}
