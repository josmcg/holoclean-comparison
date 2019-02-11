# NOTE: THIS IS A FORK OF THE HOLOCLEAN REPO ONLY FOR PURPOSES OF COMPARISON TO ANOTHER FRAMEWORK. DO NOT USE THIS AS YOUR HOLOCLEAN DISTRIBUTION

# Installation

In the main directory run 

`docker-compose build`

# Usage

To start the system run 

1. `docker-compose up db -d`
2. `docker-compose run holoclean bash`
3. Inside the docker container, run `source activate holo_env`
4. Ensure you are in the directory `/vol` (this is synced with your active repo)
5. `cd exp && ./run.sh`


# Editing the exp
See the argument parser to view how to set arguments
