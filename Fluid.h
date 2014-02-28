#ifndef FLUID_H
#define FLUID_H

#define MAX_PARTICLES 100

typedef struct my_vec3{
   float x, y, z;
} my_vec3;

typedef struct my_vec4{
   float x, y, z, alpha;
} my_vec4;

typedef struct Particle{
   my_vec3 position;       /* x, y, z position of particle */
   my_vec3 oldPosition     /* previous position */
   my_vec3 speed;          /* x, y, z velocity of particle */
   my_vec4 color;          /* Color of the particle RGB */
   my_vec4 oldColor        /* Previous color */
   float startTime;     /* Start time of the particle */
   float lifeTime;      /* How long it has been alive */
   float size;          /* size of particle */
   float angle;
   float weight;        /* weight of particle */
} Particle;

typedef struct ParticleSystem{
   Particle particles[MAX_PARTICLES]; /* Keeps track of all the particles */
   unsigned int numParticles;         /* Current number of particles alive */
   my_vec3 gravity;                      /* Particle System affects particles */
   my_vec3 wind;                         
   float milestone                    /* How long a particle can live */
   char* TextureFile;                 /* Name of the texture file */
} ParticleSystem;


#endif
