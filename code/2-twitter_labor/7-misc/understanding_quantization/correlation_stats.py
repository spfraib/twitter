import pandas as pd
import re
import os

results_dict = {
    'is_hired_1mo': {
        'iter_0-convbert-969622-evaluation':
            {
                'Spearman correlation (from scipy)': 0.8881395579335248,
                'Spearman correlation (rank + pearson)': 0.8722672008189396,
                'Kendall Tau': 0.7197876791319032, },
        'iter_1-convbert-3050798-evaluation':
            {
                'Spearman correlation (from scipy)': 0.9418813908524655,
                'Spearman correlation (rank + pearson)': 0.876934239017211,
                'Kendall Tau': 0.8035317642514531, },
        'iter_2-convbert-3134867-evaluation':
            {
                'Spearman correlation (from scipy)': 0.9611990748403862,
                'Spearman correlation (rank + pearson)': 0.9499413047821146,
                'Kendall Tau': 0.834064030085159, },
        'iter_3-convbert-3174249-evaluation':
            {
                'Spearman correlation (from scipy)': 0.958269870783558,
                'Spearman correlation (rank + pearson)': 0.8602921894475768,
                'Kendall Tau': 0.8198727327682079, },
        'iter_4-convbert-3297962-evaluation':
            {
                'Spearman correlation (from scipy)': 0.95843971223043,
                'Spearman correlation (rank + pearson)': 0.9462257553119972,
                'Kendall Tau': 0.845133425774039, }},
        'lost_job_1mo': {
            'iter_0-convbert-969622-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.8335568139557733,
                    'Spearman correlation (rank + pearson)': 0.823666532295861,
                    'Kendall Tau': 0.6489172966935665, },
            'iter_1-convbert-3050798-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.7960101709443972,
                    'Spearman correlation (rank + pearson)': 0.8080540496999495,
                    'Kendall Tau': 0.6115687108964616, },
            'iter_2-convbert-3134867-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.7510202754207926,
                    'Spearman correlation (rank + pearson)': 0.7149091105100974,
                    'Kendall Tau': 0.5721910632937749, },
            'iter_3-convbert-3174249-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.6644533395988803,
                    'Spearman correlation (rank + pearson)': 0.5215798158954073,
                    'Kendall Tau': 0.5219749221751988, },
            'iter_4-convbert-3297962-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.7247006503287751,
                    'Spearman correlation (rank + pearson)': 0.7212026276293366,
                    'Kendall Tau': 0.5460467294022979, }},

        'is_unemployed': {
            'iter_0-convbert-969622-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.6487289131825646,
                    'Spearman correlation (rank + pearson)': 0.6192423266057655,
                    'Kendall Tau': 0.4726271720448033, },
            'iter_1-convbert-3050798-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.7893726252771902,
                    'Spearman correlation (rank + pearson)': 0.7315876731902204,
                    'Kendall Tau': 0.6045011372016271, },
            'iter_2-convbert-3134867-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.9341136674863462,
                    'Spearman correlation (rank + pearson)': 0.8706793955737966,
                    'Kendall Tau': 0.7866169491351097, },
            'iter_3-convbert-3174249-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.8858158994265896,
                    'Spearman correlation (rank + pearson)': 0.8621206722001078,
                    'Kendall Tau': 0.7155938475713911, },
            'iter_4-convbert-3297962-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.9480148568018644,
                    'Spearman correlation (rank + pearson)': 0.9314851969250192,
                    'Kendall Tau': 0.8119164235910493, }},
        'job_offer': {
            'iter_0-convbert-969622-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.5098399100738237,
                    'Spearman correlation (rank + pearson)': 0.5199680298102625,
                    'Kendall Tau': 0.36445742264986974, },
            'iter_1-convbert-3050798-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.4926662556043933,
                    'Spearman correlation (rank + pearson)': 0.5254699086138541,
                    'Kendall Tau': 0.3626995642026401, },
            'iter_2-convbert-3134867-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.7758086052569864,
                    'Spearman correlation (rank + pearson)': 0.778984462516823,
                    'Kendall Tau': 0.5854864633333632, },
            'iter_3-convbert-3174249-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.5285629513933503,
                    'Spearman correlation (rank + pearson)': 0.5516072779301451,
                    'Kendall Tau': 0.3714867991863706, },
            'iter_4-convbert-3297962-evaluation':
                {
                    'Spearman correlation (from scipy)': 0.3636808462172127,
                    'Spearman correlation (rank + pearson)': 0.3738478073711762,
                    'Kendall Tau': 0.2574545121358842}},

    'job_search': {
        'iter_0-convbert-969622-evaluation':
            {
                'Spearman correlation (from scipy)': 0.8915873469903887,
                'Spearman correlation (rank + pearson)': 0.90966641601594530,
                'Kendall Tau': 0.7246217575984779},
        'iter_1-convbert-3050798-evaluation':
            {
                'Spearman correlation (from scipy)': 0.894818914532973,
                'Spearman correlation (rank + pearson)': 0.86377146910239980,
                'Kendall Tau': 0.7267431161587558},
        'iter_2-convbert-3134867-evaluation':
            {
                'Spearman correlation (from scipy)': 0.8066928813940129,
                'Spearman correlation (rank + pearson)': 0.76198185432206310,
                'Kendall Tau': 0.64483390946065},
        'iter_3-convbert-3174249-evaluation':
            {
                'Spearman correlation (from scipy)': 0.9498560156449153,
                'Spearman correlation (rank + pearson)': 0.94344930820269180,
                'Kendall Tau': 0.8065723498281273},
        'iter_4-convbert-3297962-evaluation':
            {
                'Spearman correlation (from scipy)': 0.9213685368703513,
                'Spearman correlation (rank + pearson)': 0.92356720661093440,
                'Kendall Tau': 0.7619334454051967}
    }}

results_df = pd.DataFrame.from_dict(results_dict).T
results_list = list()
for inference_folder in results_df.columns:
    print(inference_folder)
    results_iter_df = results_df[inference_folder].apply(pd.Series)
    iter_number = int(re.findall('iter_(\d)', inference_folder)[0])
    results_iter_df['iter'] = iter_number
    results_list.append(results_iter_df)
    # print(results_iter_df)
results_df = pd.concat(results_list).reset_index()
results_df = results_df.sort_values(by=['index', 'iter']).reset_index(drop=True)

# output_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/debugging/rank_correlation_torch_quantized'
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
results_df = results_df[['iter', 'index', 'Kendall Tau']]
print(results_df.head)
results_df.to_csv('kendall_tau_top10k.csv', index=False)
#results_df.to_csv(os.path.join(output_path, f'{args.country_code}.csv'), index=False)
