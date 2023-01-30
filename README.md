# ExMaps: Long-Term Localization in Dynamic Scenes using Exponential Decay

## Instructions (2023)

The files you need to run are:

1. `get_cmu_data.py`
   1. Read the first argument comments in the file to understand what it does
   2. There is code that downloads and extract all CMU slices

2. `get_coop_data.py`
   1. Read the first argument comments in the file to understand what it does
   2. This is using previous data, so only the decay is applied and results are generated

3. `get_lamar_data.py`
   1. Read the first argument comments in the file to understand what it does
   2. You have to download the data
   3. You need to specify a folder for all the data to be extracted to

For all the above the flag `do_matching` needs to be set to 1 the first time. Then when you
run the code again you can set it to 0. This will save you time.

### Results

Results are generated in the `results` folder of each dataset. The results are saved in a `.csv` file.
At each run the result `.csv` file from the `results` folder is copied to the parent folder, which is the
dataset folder. This is done to keep track of the results.
Then a file called `evaluation_results_2022_aggregated.csv` is generated from the 3 (or N) runs. 
This is the file to read from. At this point you can parse the results and generate the graphs.
Do not run `analyse_results_models_*.py` or `main.py` or `get_lamar/cmu/coop_data.py` again.

## Below are the old instructions (pre 2023)

### This repo contains the code for the above paper accepted in WACV2021.

The basic commands to run for generating the results are:

    python3 get_visibility_matrix.py /home/user/fullpipeline/colmap_data/CMU_data/slice3/ 
    python3 get_points_3D_mean_descs.py /home/user/fullpipeline/colmap_data/CMU_data/slice3/ 
    python3 main.py /home/user/fullpipeline/colmap_data/CMU_data/slice3/

The directory `/home/user/fullpipeline/colmap_data/CMU_data/slice3/` will be different in your case. 

`get_visibility_matrix.py` applies the exponential decay.

Once you run this code for all the CMU slices (or retail shop) then you will want to run `results_analyzer.py`. 

#### Preparation of data

Before you ran `get_visibility_matrix.py` and the other commands you will need to have setup your data.
For this project I did it manually.

You will need to seperate your sessions. 
If you have 9 slices for example, you can pick slice 1 for your base model, and slice 9 for your query (test session).

You will need to create thus 10 folders.

- slice1
- ...
- slice1_9

each session folder should look like this:

```bash
├── base
│   ├── base_images_cam_centers.txt
│   ├── database.db
│   ├── images
│   │   ├── img_01007_c0_1303398554379838us.jpg
│   │   ├── ...
│   └── model
│       └── 0
│           ├── cameras.bin
│           ├── images.bin
│           ├── points3D.bin
│           └── project.ini
├── gt
│   ├── database.db
│   ├── images
│   │   └── session_2
│   │       ├── img_02455_c0_1283347962184884us.jpg
│   │       ├── ...
│   ├── model
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   └── points3D.bin
│   └── query_name.txt
├── live
│   ├── database.db
│   ├── images
│   │   ├── session_3
│   │   │   ├── img_01750_c0_1284563401010858us.jpg
│   │   │   ├── ...
│   │   └── session_4
│   │       ├── img_01915_c0_1285949838575748us.jpg
│   │       ├── ...
│   ├── model
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   └── points3D.bin
│   ├── query_name.txt
│   └── session_lengths.txt
```

The files under the models folders will be created when you run this: 

```python3 cmu_sparse_reconstuctor.py /home/user/fullpipeline/colmap_data/Coop_data/slice1/ 0```
```python3 cmu_sparse_reconstuctor.py /home/user/fullpipeline/colmap_data/Coop_data/slice1_1/ 0```
```...```
```python3 cmu_sparse_reconstuctor.py /home/user/fullpipeline/colmap_data/Coop_data/slice1_9/ 0```

The script can be found [here](https://github.com/alexs7/Mobile-Pose-Estimation-Pipeline-Prototype/blob/server_version/cmu_sparse_reconstuctor.py).

All the above ws preparation work to create the incremental live models. Once you have all these sorted you can then run the commands mentioned before.

Notes: 

 1. This repo is still under construction. For any questions please
        contact ar2056(at)bath.ac.uk.   
 2. The data will have to be added for the
        code to run. Once you do add it it is easy to replace it with yours.
