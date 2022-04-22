# Demographics of Twitter users

We use [m3inference](https://github.com/euagendas/m3inference) to infer the demographics of our Twitter users. There are several steps to follow to perform these inferences.

## Get user pictures

- Run `sbatch --array=0-SIZE_ARRAY get_user_images.sbatch MODE`. Use the biggest `SIZE_ARRAY` as possible as this will define the number of tar files you get as output. `MODE` can be either:
  - `from_scratch` if you are downloading pictures from scratch
  - `get_missing` in case you have already downloaded pictures and want to retry downloading the pictures you have not been able to download yet
- Run `sbatch get_filenames_from_tars.sbatch`. This will go through all tar files containing pictures and list the file names as well as errors in separate text files.
- Run `sbatch identify_missing_pictures.sbatch`. This will give you a summary on the pictures you have downloaded. All of the info will be logged in the `.out` file resulting from the batch launch.

## Perform demographic inference

Once you have the pictures and you are sure you are not missing any, you can perform the inference:
- Run `sbatch generate_user_image_map.sbatch`. This will create new user files containing the tar path where their image is stored.
- Run `sbatch --array=0-SIZE_ARRAY demographic_inference COUNTRY_CODE`. Options for `COUNTRY_CODE` are:
  - `all` if you would like to perform inference for users in all countries for which we have data
  - a specific country code (e.g. `US`) to do inference only for the users in that country