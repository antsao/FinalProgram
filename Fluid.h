#ifndef FLUID_H
#define FLUID_H

#define MAX_PARTICLES 100

typedef struct Particle{
   vec3 position;       /* x, y, z position of particle */
   vec3 oldPosition     /* previous position */
   vec3 speed;          /* x, y, z velocity of particle */
   vec3 color;          /* Color of the particle RGB */
   vec3 oldColor        /* Previous color */
   float startTime;     /* Start time of the particle */
   float lifeTime;      /* How long it has been alive */
   float size;          /* size of particle */
   float weight;        /* weight of particle */
} Particle;

typedef struct ParticleSystem{
   Particle particles[MAX_PARTICLES]; /* Keeps track of all the particles */
   unsigned int numParticles;         /* Current number of particles alive */
   vec3 gravity;                      /* Particle System affects particles */
   vec3 wind;                         
   float milestone                    /* How long a particle can live */
   char* TextureFile;                 /* Name of the texture file */
} ParticleSystem;


#endif
