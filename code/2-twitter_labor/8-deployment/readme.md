## How to run inference:

### Run a single job as a test:
`sbatch inference_ONNX_bert_MX.sbatch MX 6 3 mexrun_5GB 0`

### Run a job-array:
`sbatch --array=0-1950 inference_ONNX_bert_MX.sbatch MX 6 3 mexrun_5GB 0`

### Check how many files are left to run in a job:
`sbatch check_remaining_files.sbatch MX 6 3 mexrun_5GB 0`

The output will be stored in `outfiles/`

For example:

```
/scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/MX/
/scratch/spf248/twitter/data/user_timeline/user_timeline_BERT_scores_labor/MX/iter_6--3GB-
/scratch/spf248/twitter/data/user_timeline/user_timeline_BERT_scores_labor/MX/iter_6--3GB-/logs
/scratch/spf248/twitter/data/user_timeline/user_timeline_BERT_scores_labor/MX/iter_6--3GB-/output
pyenv activated
/scratch/spf248/twitter/code/twitter/code/2-twitter_labor/8-deployment
SLURM_ARRAY_TASK_ID=test
running inference..
python -u check_remaining_files.py --input_path /scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/MX/ --output_path /scratch/spf248/twitter/data/user_timeline/user_timeline_BERT_scores_labor/MX/iter_6--3GB-/output --country_code MX --iteration_number 6 --method 0 --resume 0
done
libs loaded
Namespace(country_code='MX', debug_mode=True, drop_duplicates=False, input_path='/scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/MX/', iteration_number=6, method=0, output_path='/scratch/spf248/twitter/data/user_timeline/user_timeline_BERT_scores_labor/MX/iter_6--3GB-/output', resume=0)
Load random Tweets:
resume: 0
 files already run 2966
 original number of files to run 20000
 files to run after resume: 20000
1.53user 0.34system 0:01.90elapsed 98%CPU (0avgtext+0avgdata 226072maxresident)k
0inputs+8outputs (0major+100638minor)pagefaults 0swaps
```

This tells you that 2966 files (see inside of `check_remaining_files.sbatch` for `OUTPUT_PATH`) have already been processed, and the input folder (see `INPUT_PATH`) has 20000 files. However, you see that files to run after resume is still 20000 files. This is because none of the unique ids in any of the 2966 already processed files match the unique ids in the 20000 files inside `output_path`). 

### Debug help: To run a single job as a python command:
`python -u check_remaining_files.py --input_path /scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/MX/ --output_path /scratch/spf248/twitter/data/user_timeline/user_timeline_BERT_scores_labor/MX/iter_6--3GB-/output --country_code MX --iteration_number 6 --method 0 --resume 0`
Note: the sbatch files print out the python command that was run. You can copy paste this for running. 

## Note: 
The unique file ids are extracted from the filenames according to a pattern. So if you change how files are stored, please update the pattern. 

Refer to this code block in the code:
```
"""
creating a list of unique file ids assuming this file name structure:
/scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/part-00000-52fdb0a4-e509-49fe-9f3a-d809594bba7d-c000.snappy.parquet
in this case:
unique_intput_file_id_list will contain 00000-52fdb0a4-e509-49fe-9f3a-d809594bba7d
filename_prefix is /scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/part-
filename_suffix is -c000.snappy.parquet
"""
unique_intput_file_id_list = [filename.split('part-')[1].split('-c000')[0]
                              for filename in input_files_list]
filename_prefix = input_files_list[0].split('part-')[0]
filename_suffix = input_files_list[0].split('part-')[1].split('-c000')[1]
```
