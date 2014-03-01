#include "ParticleUpdate.h"
#include "Fluid.h"
#include <stdio.h>

void updateParticles(Particle *particles, int size, int force, int mode) {
  dim3 dimBlock(250);
  dim3 dimGrid(50);

  my_vec3 localExtForce;
  if (force == GRAVITY) {
    printf("making gravity\n");
    localExtForce.x = 0;
    localExtForce.y = -3;
    localExtForce.z = 0;
  }

  Particle *d_particles;
  cudaMalloc(&d_particles, sizeof(Particle) * size);
  cudaMemcpy(d_particles, particles, sizeof(Particle) * size, cudaMemcpyHostToDevice);

  my_vec3 *d_localExtForce;
  cudaMalloc(&d_localExtForce, sizeof(my_vec3));
  cudaMemcpy(d_localExtForce, &localExtForce, sizeof(my_vec3), cudaMemcpyHostToDevice);

  if (mode == NO_COLLISION) {
    printf("right before making call\n");
    updateParticleKernel<<<dimBlock, dimGrid>>>(particles, localExtForce);
  }
  printf("right after making call\n");

  cudaMemcpy(particles, d_particles, sizeof(Particle) * size, cudaMemcpyDeviceToHost);

  cudaFree(d_particles);
  cudaFree(d_localExtForce);
}

__global__ void updateParticleKernel(Particle *particles, my_vec3 extForce) {
  int pInd = blockIdx.x * blockDim.x + threadIdx.x;
  float deltaTime = 0.01;
  Particle part = particles[pInd];

  part.velocity.x = part.velocity.x + extForce.x * deltaTime;
  part.velocity.y = part.velocity.y + extForce.y * deltaTime;
  part.velocity.z = part.velocity.z + extForce.z * deltaTime;

  part.position.x = part.position.x + part.velocity.x * deltaTime;
  part.position.y = part.position.y + part.velocity.y * deltaTime;
  part.position.z = part.position.z + part.velocity.z * deltaTime;
}
