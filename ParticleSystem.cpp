#include "Fluid.h"
// MEMBERS
//      Particle particles[MAX_PARTICLES]; /* Keeps track of all the particles */
//      unsigned int numParticles;         /* Current number of particles alive */
//      my_vec3 gravity;                      /* Particle System affects particles */
//      my_vec3 wind;                         
//      float milestone;                    /* How long a particle can live */
//      char* TextureFile;                 /* Name of the texture file */

//Particle System Functions/
ParticleSystem::ParticleSystem(my_vec3 gravity, my_vec3 wind, float milestone, char* Texture) :
   gravity(gravity),
   wind(wind),
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
    if (particleCount % 101 == 0) {
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
    particles[i].velocity.x = 0;
    particles[i].velocity.y = -(rand()/(float)RAND_MAX) + rand()%20 + 3;
    particles[i].velocity.z = 0;
    particleCount++;
  }
   
}

void ParticleSystem::update(float deltaTime) {
   #ifdef ENABLE_CUDA
     updateParticles(particles, NUM_PARTICLES, GRAVITY);    
   #else
      //Implement CPU Version
   #endif
}
