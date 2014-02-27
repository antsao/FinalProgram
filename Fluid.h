#ifndef FLUID_H
#define FLUID_H

#define MAX_PARTICLES 100

typedef struct Particle{
   vec3 position;       /* x, y, z position of particle */
   vec3 speed;          /* x, y, z velocity of particle */
   vec3 color;          /* Color of the particle RGB */
   float startTime;     /* Start time of the particle */
   float lifeTime;      /* How long it has been alive */
} Particle;

typedef struct ParticleSystem{
   Particle particles[MAX_PARTICLES]; 
   vec3 gravity         /* Particle System affects particles */
   float milestone      /* How long a particle can live */
} ParticleSystem;


#endif
