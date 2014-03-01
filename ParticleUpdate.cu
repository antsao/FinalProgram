#include "ParticleUpdate.h"

Particle *updateParticles(Particle *particles, int size) {
   dim3 dimBlock(1024);
   dim3 dimGrid(1024);

   my_vec3 localExtForce;
   localExtForce.x = 0;
   localExtForce.y = -3;
   localExtForce.z = 0;

   Particle *d_particles;
   cudaMalloc(&d_particles, sizeof(Particle) * size);
   cudaMemcpy(d_particles, particles, sizeof(Particle) * size, cudaMemcpyHostToDevice);

   my_vec3 *d_localExtForce;
   cudaMalloc(&d_localExtForce, sizeof(my_vec3));
   cudaMemcpy(d_localExtForce, &localExtForce, sizeof(my_vec3), cudaMemcpyHostToDevice);

   updateParticleKernel<<<dimBlock, dimGrid>>>(particles, localExtForce);

   cudaMemcpy(particles, d_particles, sizeof(Particle) * size, cudaMemcpyDeviceToHost);

   cudaFree(d_particles);
   cudaFree(d_localExtForce);
   return particles;
}

__global__ void updateParticleKernel(Particle *particles, my_vec3 extForce) {
   int pInd = blockIdx.x * threadIdx.x;
   float deltaTime = 0.01;
   Particle part = particles[pInd];

   part.velocity.x = part.velocity.x + extForce.x * deltaTime;
   part.velocity.y = part.velocity.y + extForce.y * deltaTime;
   part.velocity.z = part.velocity.z + extForce.z * deltaTime;

   part.position.x = part.position.x + part.velocity.x * deltaTime;
   part.position.y = part.position.y + part.velocity.y * deltaTime;
   part.position.z = part.position.z + part.velocity.z * deltaTime;
}
