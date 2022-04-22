## Run the application from a docker container
1. Install docker (https://www.docker.com/).

2. Open a terminal, run `docker pull ghcr.io/schapirolabor/mcmicrogui:latest` and wait for the download to finish.

3. Run `docker images` to see the ID of the pulled image.

4. Run `docker run -d -t -v <YOUR PATH TO MC MCMICRO DATA>:/input -v <PATH TO SAVE THE REPORT OF MCMCIRO GUI>:/output <DOCKER IMAGE ID>`. 
   Replace the text in <> with the appropriate values/paths.

5. Run `docker ps` to see the name of the container.

6. Run `docker exec -it <CONTAINER NAME> /bin/bash` to open a shell inside the container.

7. Run `python mcmicrogui/app.py -i input -o output -m <YOUR MARKER>`
   <YOUR MARKER> is the marker you want to visualize on the report.

8. The report should now generate. If you find any bugs or have suggestions for new features, report them here: https://github.com/SchapiroLabor/mcmicrogui/issues



## Run the application from the source code

**

1. Install Python 3.10.

2. Download or clone the repository.

3. Run MCMICRO on the data of your choice. Make sure the output ends up in the data folder in the downloaded repository.

4. Run `pip install -r requirements.txt` to install all required packages

5. Run `python app.py` to launch the script.

6. If you find any bugs or have suggestions for new features, report them here: https://github.com/SchapiroLabor/mcmicrogui/issues
