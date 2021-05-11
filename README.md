# System Execution Modelling (SEM)

This repository contains sample models and visualisation software intended for use with the University of Adelaide's MEDEA/re software environment, created by the Centre  for Distributed and Intelligent Technologies - Modelling and Analysis (CDIT-MA). MEDEA/re can be downloaded from the following github site:

https://github.com/cdit-ma/SEM

Before using the cdcin-trc/SEM software models and visualisation, ensure you:

1 - Set up the backend environment (re/Jenkins) following instructions from https://github.com/cdit-ma/SEM

2 - Set up the modelling environment (MEDEA) following instructions from https://github.com/cdit-ma/SEM
  - for now, you can ignore the worker definitions. This is discussed further below

Once this is done,

3 - Set up visualisation environment using the "medeaviz" tool.
4 - Decide which MEDEA models you want to deploy
5 - Determine which "workers" are needed for these models
5 - Configure the /medeaworkers/deployment/CMakeLists.txt file, commenting out workers you do not want to deploy, leaving only the workers to deploy
6 - If you have changed the /medeaworkers/deployment/CMakeLists.txt, commit and push the changes via git. This is needed for Jenkins in the next step
6b - configure your local graphml worker definitions according to that
7 - Execute the Jenkins deploy_worker job to compile the workers
8 - Execute the Scenario Manager
  - this runs as a MEDEA model, so execute this from within MEDEA
  - if successful, you should see state updates in the MEDEA console, depending on your log verbosity level
9 - Execute the medeaviz visualiser
  - if successful, you should see:
    - a visual state update, showing surface/subsurface "contacts" moving, and the clock increasing
10 - Execute models
11 - Ensure the models are working by looking at the console or medeaviz

