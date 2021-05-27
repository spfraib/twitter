#SBATCH --job-name=Demographic_Inference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15GB
#SBATCH --time=24:00:00

module purge
module load python3/intel/3.6.3
source /beegfs/vp1274/twitter/venvs/twitter-env/bin/activate
cd /home/vp274/

python demographic_inference.py --images_dir "/scratch/spf248/twitter/data/classification/US/profile_pictures_sam/" \
--resized_images_dir "/scratch/spf248/twitter/data/classification/US/profile_pictures_resized_random/" \
--output_dir "/scratch/spf248/twitter/data/classification/US/inference_results" \
--country "US"