# System Execution Modelling (SEM)

This repository contains sample models and visualisation software intended for use with the University of Adelaide's MEDEA/re software environment, created by the Centre  for Distributed and Intelligent Technologies - Modelling and Analysis (CDIT-MA). MEDEA/re can be downloaded from the following github site:

https://github.com/cdit-ma/SEM

<H2>Installation and Configuration Instructions</H2>

<H3>Step 1 - Set up the RE / Jenkins Backend Environment</H3>
Set up the backend environment (re/Jenkins) following instructions from https://github.com/cdit-ma/SEM

<H3>Step 2 - Set up MEDEA </H3>
Set up the modelling environment (MEDEA) following instructions from https://github.com/cdit-ma/SEM

For now, you can ignore the worker definitions. This is discussed further below

<H3>Step 3 - Setup MEDEAVIZ </H3>
Set up visualisation environment using the "medeaviz" tool.

<H3>Step 4 - Select Models </H3>
Three models have been provided

- An Optronics Model
- An ESM Model
- A Particle FIletr based Tracking Algorithm 

Decide which MEDEA models you want to deploy

Note: The Adaptive Swarm Optimized Particle Filter is included to demonstrate some advanced capabilities, and should not be used in your first experiment.

<H3>Step 5 - Install Custom Workers</H3>
Determine which "workers" are needed for these models

<H3>Step 6</H3>
Configure the /medeaworkers/deployment/CMakeLists.txt file, commenting out workers you do not want to deploy, leaving only the workers to deploy

<H3>Step 7</H3>
If you have changed the /medeaworkers/deployment/CMakeLists.txt, commit and push the changes via git. This is needed for Jenkins in the next step

<H3>Step 8</H3>
Configure your local graphml worker definitions according to that

<H3>Step 9</H3>
<H4> Step 9.1 </H4>
Configure jenkins to add the deploy_worker task. 

- copy the file SEM/jenkins/deploy_workers.groovy to the jenkinns server.

- create a jenkins task

- TODO how do we point this at the SEM git repo.

<H4> Step 9.2 </H4>
Execute the Jenkins deploy_worker job to compile the workers

<H3>Step 10</H3>
Execute the Scenario Manager

<p>This runs as a MEDEA model, so execute this from within MEDEA

- if successful, you should see state updates in the MEDEA console, depending on your log verbosity level

<H3>Step 11</H3>
Execute the medeaviz visualiser

- if successful, you should see:
  - a visual state update, showing surface/subsurface "contacts" moving, 
  - and the clock increasing

<H3>Step 12</H3>
Execute models

<H3>Step 13</H3>Ensure the models are working by looking at the console or medeaviz

<H3> Step 14 </H4>
This is an optional step if you wish to attempt to run some of the advanced modesl such as the Adaptive Swarm Optimized Particle Filter. These models make use of heavily optimized libraries like blas and fftw, as well as their GPU equivalents cufft and cublas; to allow reliable comparison bewteen cpu and gpu performance. 

Before executing these models you must install the third party libraries

- http://math-atlas.sourceforge.net/
- https://developer.nvidia.com/cufft
- http://www.fftw.org/download.html
- https://developer.nvidia.com/cufft

and update the CMakeList files to referfence where you have installed them, before rebuilding the workers.
