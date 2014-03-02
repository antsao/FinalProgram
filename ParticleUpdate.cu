#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "ParticleUpdate.h"
#include "Fluid.h"

void updateParticles(Particle *particles, int size, int force, int mode) {
  dim3 dimBlock(256);
  dim3 dimGrid(32);

  my_vec3 localExtForce;
  if (force == GRAVITY) {
    localExtForce.x = 0;
    localExtForce.y = -3;
    localExtForce.z = 0;
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
  cudaMalloc(&d_localExtForce, sizeof(my_vec3));
  if (cudaMemcpy(d_localExtForce, &localExtForce, 
                 sizeof(my_vec3), 
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("didn't copy force\n");
  }

  if (mode == NO_COLLISION) {
    updateParticleKernel<<<dimGrid, dimBlock>>>(particles, localExtForce);
  }
  cudaMemcpy(particles, d_particles, sizeof(Particle) * size, cudaMemcpyDeviceToHost);

  cudaFree(d_particles);
  cudaFree(d_localExtForce);
}

__global__ void updateParticleKernel(Particle *particles, my_vec3 extForce) {
  float deltaTime = 0.01;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  Particle part = particles[idx];

  part.velocity.x = part.velocity.x + extForce.x * deltaTime;
  part.velocity.y = part.velocity.y + extForce.y * deltaTime;
  part.velocity.z = part.velocity.z + extForce.z * deltaTime;

  part.position.x = part.position.x + part.velocity.x * deltaTime;
  part.position.y = part.position.y + part.velocity.y * deltaTime;
  part.position.z = part.position.z + part.velocity.z * deltaTime;

  particles[idx] = part;
}
