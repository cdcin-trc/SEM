# System Execution Modelling (SEM)

This repository contains sample models and visualisation software intended for use with the University of Adelaide's MEDEA/re software environment, created by the Centre  for Distributed and Intelligent Technologies - Modelling and Analysis (CDIT-MA). MEDEA/re can be downloaded from the following github site:

https://github.com/cdit-ma/SEM

Before using the software on the (cdcin-trc) site, ensure you:

1 - Setup the backend environment (re/Jenkins) following instructions from https://github.com/cdit-ma/SEM
2 - Setup the modelling environment (MEDEA) following instructions from https://github.com/cdit-ma/SEM
  - for now, you can ignore the worker definitions. This is discussed further below

Once this is done, 
- set up visualisatio environment
  - medeaviz
- setup jenkins task deploy_worker  
  - obtained from https://github.com/cdcin-trc/SEM
- set up working models:
  - scenario manager
     - ensure it's working via medeaviz
- decide on set of models
  - configure cmake according to that
  - configure local graphml worker definitions according to that
  - deploy_workers to compile workers on back end
- run models
- ensure models are working by looking at medeaviz

