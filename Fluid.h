#ifndef FLUID_H
#define FLUID_H

#define MAX_PARTICLES 100
#define GRAVITY 1
#define NO_COLLISION 2

typedef struct my_vec3{
   float x, y, z;
} my_vec3;

typedef struct my_vec4{
   float x, y, z, alpha;
} my_vec4;

typedef struct Particle{
   my_vec3 position;       /* x, y, z position of particle */
   my_vec3 velocity;          /* x, y, z velocity of particle */
} Particle;

typedef struct ParticleSystem{
   Particle particles[MAX_PARTICLES]; /* Keeps track of all the particles */
   unsigned int numParticles;         /* Current number of particles alive */
   my_vec3 gravity;                      /* Particle System affects particles */
   my_vec3 wind;                         
   float milestone;                    /* How long a particle can live */
   char* TextureFile;                 /* Name of the texture file */
} ParticleSystem;


#endif
