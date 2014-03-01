#ifndef PARTICLE_UPDATE_H
#define PARTICLE_UPDATE_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Fluid.h"

Particle *updateParticles(Particle *particles, int size);
__global__ void updateParticleKernel(Particle *particles, my_vec3 extForce);

#endif
