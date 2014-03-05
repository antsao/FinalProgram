#include <stdio.h>
#include "Fluid.h"
#include "ParticleUpdate.h"

// MEMBERS
//      Particle particles[MAX_PARTICLES]; /* Keeps track of all the particles */
//      unsigned int numParticles;         /* Current number of particles alive */
//      my_vec3 gravity;                      /* Particle System affects particles */
//      my_vec3 wind;                         
//      float milestone;                    /* How long a particle can live */
//      char* TextureFile;                 /* Name of the texture file */

//Particle System Functions/
ParticleSystem::ParticleSystem(float milestone, char* Texture) :
   milestone(milestone),
   TextureFile(Texture)
{}

void ParticleSystem::initalize()
{
  srand(time(NULL));
  
  float particleYPos = 25;
  float particleXPos = -25;
  float particleZPos = 25;
  int particleCount = 1;

  for (int i = 0; i < NUM_PARTICLES; i++) {
    if (particleCount % 51 == 0) {
      particleZPos = 25;
      if (particleXPos != 25) {
        particleXPos++;
      }
      else {
        particleXPos = -25;
        particleYPos--;
        particleZPos = 25;
      }
    }
    else {
      particleZPos--;
    }
    particles[i].position.x = particleXPos;
    particles[i].position.y = particleYPos;
    particles[i].position.z = particleZPos;
    particles[i].velocity.x = 3 - rand() % 5;//(((rand()%2)?1:-1))*(rand()/(float)RAND_MAX) + rand() % 20 + 5;
    particles[i].velocity.y = 2 - rand() % 5;//(((rand()%2)?1:-1))*(rand()/(float)RAND_MAX) + rand() % 20 + 5;
    particles[i].velocity.z = 1 - rand() % 5;//(((rand()%2)?1:-1))*(rand()/(float)RAND_MAX) + rand() % 20 + 5;
    particles[i].color.x = (float)((rand()/(float)RAND_MAX) + rand()%3)/10;
    particles[i].color.y = (float)((rand()/(float)RAND_MAX) + rand()%3)/10;
    particles[i].color.z = 1;//(float)(rand()/(float)RAND_MAX);
    particleCount++;
  }
   
}

void ParticleSystem::update(float deltaTime) {
  #ifdef ENABLE_CUDA
    my_vec3 gravity;
    gravity.x = 0.0f;
    gravity.y = -3.0f;
    gravity.z = 0.0f;

    my_vec3 wind;
    wind.x = 5.0f;
    wind.y = 0.0f;
    wind.z = 0.0f; 

    updateParticles(particles, NUM_PARTICLES, gravity);    
  #else
  //Implement CPU Version
  #endif
}
