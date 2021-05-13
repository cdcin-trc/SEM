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
Set up the visualisation environment using the "medeaviz" tool. This will allow live viewing of results and scenarios, but is not strictly required.

<H3>Step 4 - Select Models </H3>
Two models have been provided:

- an Optronics Model, and
- a Particle Filter based Tracking Algorithm 

In addition, there is a Scenario Manager model which is used to modulate the workload generated by the above models during execution. The Partile Filter model has its own manager which collaborates with the Scenario manager to modulate its workload.

Decide which MEDEA models you want to deploy.

Note: The Adaptive Swarm Optimized Particle Filter is included to demonstrate some advanced capabilities, and should not be used in your first experiment.

<H3>Step 5 - Install Custom Workers</H3>
MEDEA models are usually primarily designed using the in-built visual language. However, MEDEA models can be augmented by using "workers", which are large or complex blocks of code not amenable to visual design, e.g. a Fast-Fourier Transform. MEDEA comes with several built-in workers; the https://github.com/cdcin-trc/SEM repository extends this basic worker set considerably. Some models require specific workers to function correctly. Workers needed for specific models are:

- Scenario Manager
  - ScenarioManagerWorker
- Optronics
  - ScenarioManagerWorker
  - CpuStatWorker
  - OptronicsWorker
  - TrackingWorker
- Particle Filter
  - ScenarioManagerWorker
  - AdaptiveSwarmOptimisedPartocleFilterWorker

<H3>Step 6 - Update CMakeLists</H3>
The `/medeaworkers/deployment/CMakeLists.txt` file configures which workers will be compiled and made available to the MEDEA models upon execution. Make changes to this file, commenting out workers you do not want to deploy, leaving only the workers you want to deploy.

<H3>Step 7 - Push medeaworkers to git</H3>
If you have changed the `/medeaworkers/deployment/CMakeLists.txt`, commit and push the changes via git to your repositry. This is needed for Jenkins in the next step.

<H3>Step 8 - Update Local Worker Definition Files</H3>
Copy the relevant graphml worker definitions from the `/medeaworkers/modelling/WorkerDefinitions` folder to your local MEDEA installation directory.

<H3>Step 9 - Deploy The Workers</H3>
<H4>Step 9.1</H4>
Configure Jenkins to add the deploy_worker task. 

- copy the file `SEM/jenkins/deploy_workers.groovy` to the Jenkins server.
- create a Jenkins task
- point the task at your MEDEA workers git repo

Note: you should only need to perform this step once.

<H4> Step 9.2 </H4>
Execute the Jenkins deploy_worker job to compile the workers. This step should be repeated whenever you make changes to any of the workers in your git repo.

<H3>Step 10 - Execute the Scenario Manager</H3>
This runs as a MEDEA model, so execute this from within MEDEA
- you will need to point MEDEA at your Jenkins environment
- you will also need to reassign the assembly in the Hardware view to match your local hardware configuration.
- if successful, you should see state updates in the MEDEA console, depending on your log verbosity level.

<H3>Step 11 - Run Medeaviz</H3>
Execute the medeaviz visualiser

- if successful, you should see:
  - a visual state update, showing surface/subsurface "contacts" moving and
  - the clock increasing

<H3>Step 12 - Execute Models</H3>
- you will need to point MEDEA at your Jenkins environment
- you will also need to reassign the assembly in the Hardware  view to match your local hardware configuration.
- if successful, you should see state updates in the MEDEA console, depending on your log verbosity level

<H3>Step 13 - Verify Model Execution</H3>
Ensure the models are working by looking at the console or medeaviz

<H3> Step 14 - Advanced Model Configuration</H4>
This is an optional step if you wish to attempt to run some of the advanced models such as the Adaptive Swarm Optimized Particle Filter. These models make use of heavily optimized libraries like blas and fftw, as well as their GPU equivalents cufft and cublas; to allow reliable comparison bewteen CPU and GPU performance. 

Before executing these models you must install the third party libraries:

- http://math-atlas.sourceforge.net/
- https://developer.nvidia.com/cufft
- http://www.fftw.org/download.html
- https://developer.nvidia.com/cufft

and update the CMakeList files to reference where you have installed them, before rebuilding the workers.
