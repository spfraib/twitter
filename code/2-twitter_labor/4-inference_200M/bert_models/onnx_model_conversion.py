# code to convert saved pytorch models to optimized+quantized onnx models for faster inference

import os
import shutil
from transformers.convert_graph_to_onnx import convert, optimize, quantize, verify
import argparse
import logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_model_path_to_model_name(model_path):
    if 'bertweet' in model_path:
        return 'vinai/bertweet-base'
    elif 'roberta-base' in model_path:
        return 'roberta-base'
    elif 'DeepPavlov' in model_path:
        return 'DeepPavlov/bert-base-cased-conversational'


parser = argparse.ArgumentParser()

parser.add_argument("--iteration_number", type=int, help="path to pytorch models (with onnx model in model_path/onnx/")
parser.add_argument("--country_code", type=str)
parser.add_argument("--method", type=int)

args = parser.parse_args()

if args.method == 0:
    best_model_paths_dict = {
        'US': {
            'iter0': {
                'lost_job_1mo': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928497_SEED_14',
                'is_hired_1mo': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928488_SEED_5',
                'is_unemployed': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928498_SEED_15',
                'job_offer': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928493_SEED_10',
                'job_search': 'DeepPavlov_bert-base-cased-conversational_jan5_iter0_928486_SEED_3'
                # 'lost_job_1mo': 'vinai_bertweet-base_jan5_iter0_928517_SEED_7',
                # 'is_hired_1mo': 'vinai_bertweet-base_jan5_iter0_928525_SEED_15',
                # 'is_unemployed': 'vinai_bertweet-base_jan5_iter0_928513_SEED_3',
                # 'job_offer': 'vinai_bertweet-base_jan5_iter0_928513_SEED_3',
                # 'job_search': 'vinai_bertweet-base_jan5_iter0_928513_SEED_3'
            },
            'iter1': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_feb22_iter1_3045488_seed-2',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_feb22_iter1_3045493_seed-7',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_feb22_iter1_3045488_seed-2',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_feb22_iter1_3045500_seed-14',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_feb22_iter1_3045501_seed-15'},
            'iter2': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_feb23_iter2_3132744_seed-9',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_feb23_iter2_3132736_seed-1',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_feb23_iter2_3132748_seed-13',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_feb23_iter2_3132740_seed-5',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_feb23_iter2_3132741_seed-6'
            },
            'iter3': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_feb25_iter3_3173734_seed-11',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_feb25_iter3_3173731_seed-8',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_feb25_iter3_3173735_seed-12',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_feb25_iter3_3173725_seed-2',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_feb25_iter3_3173728_seed-5'
            },
            'iter4': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297481_seed-7',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297477_seed-3',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297478_seed-4',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297477_seed-3',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_mar1_iter4_3297484_seed-10'
            },
            'iter5': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may26_iter5_7147724_seed-14',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may26_iter5_7147721_seed-11',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may26_iter5_7147712_seed-2',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may26_iter5_7147715_seed-5',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may26_iter5_7147720_seed-10'
            },
            'iter6': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may29_iter6_7232047_seed-3',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may29_iter6_7232051_seed-7',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may29_iter6_7232051_seed-7',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may29_iter6_7232045_seed-1',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may29_iter6_7232059_seed-15'
            },
            'iter7': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_jul19_iter7_8788309_seed-4',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_jul19_iter7_8788311_seed-6',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_jul19_iter7_8788314_seed-9',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_jul19_iter7_8788309_seed-4',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_jul19_iter7_8788315_seed-10'
            },
            'iter8': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_jul22_iter8_8850799_seed-5',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_jul22_iter8_8850801_seed-7',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_jul22_iter8_8850800_seed-6',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_jul22_iter8_8850798_seed-4',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_jul22_iter8_8850802_seed-8'
            },
        },
        'BR': {
            'iter0': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_feb16_iter0_2843324_seed-12',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_feb16_iter0_2843317_seed-5',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_feb16_iter0_2843317_seed-5',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_feb16_iter0_2843318_seed-6',
                'job_search': 'neuralmind-bert-base-portuguese-cased_feb16_iter0_2843320_seed-8'
            },
            'iter1': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_mar12_iter1_3742968_seed-6',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_mar12_iter1_3742968_seed-6',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_mar12_iter1_3742972_seed-10',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_mar12_iter1_3742970_seed-8',
                'job_search': 'neuralmind-bert-base-portuguese-cased_mar12_iter1_3742966_seed-4'},

            'iter2': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_mar24_iter2_4173786_seed-10',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_mar24_iter2_4173783_seed-7',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_mar24_iter2_4173787_seed-11',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_mar24_iter2_4173785_seed-9',
                'job_search': 'neuralmind-bert-base-portuguese-cased_mar24_iter2_4173784_seed-8'
            },
            'iter3': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_apr1_iter3_4518519_seed-6',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_apr1_iter3_4518514_seed-1',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_apr1_iter3_4518519_seed-6',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_apr1_iter3_4518525_seed-12',
                'job_search': 'neuralmind-bert-base-portuguese-cased_apr1_iter3_4518514_seed-1'},
            'iter4': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_apr3_iter4_4677938_seed-6',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_apr3_iter4_4677933_seed-1',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_apr3_iter4_4677934_seed-2',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_apr3_iter4_4677934_seed-2',
                'job_search': 'neuralmind-bert-base-portuguese-cased_apr3_iter4_4677933_seed-1'},
            'iter5': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_may17_iter5_6886444_seed-13',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_may17_iter5_6886442_seed-11',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_may17_iter5_6886435_seed-4',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_may17_iter5_6886437_seed-6',
                'job_search': 'neuralmind-bert-base-portuguese-cased_may17_iter5_6886436_seed-5'
            },
            'iter6': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_may22_iter6_7053031_seed-4',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_may22_iter6_7053036_seed-9',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_may22_iter6_7053029_seed-2',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_may22_iter6_7053030_seed-3',
                'job_search': 'neuralmind-bert-base-portuguese-cased_may22_iter6_7053034_seed-7'
            },
            'iter7': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_may30_iter7_7267580_seed-7',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_may30_iter7_7267580_seed-7',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_may30_iter7_7267580_seed-7',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_may30_iter7_7267588_seed-15',
                'job_search': 'neuralmind-bert-base-portuguese-cased_may30_iter7_7267588_seed-15'
            },
            'iter8': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_jun8_iter8_7448808_seed-15',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_jun8_iter8_7448795_seed-2',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_jun8_iter8_7448805_seed-12',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_jun8_iter8_7448800_seed-7',
                'job_search': 'neuralmind-bert-base-portuguese-cased_jun8_iter8_7448794_seed-1'
            },
            'iter9': {
                'lost_job_1mo': 'neuralmind-bert-base-portuguese-cased_jun24_iter9_7985493_seed-11',
                'is_hired_1mo': 'neuralmind-bert-base-portuguese-cased_jun24_iter9_7985486_seed-4',
                'is_unemployed': 'neuralmind-bert-base-portuguese-cased_jun24_iter9_7985495_seed-13',
                'job_offer': 'neuralmind-bert-base-portuguese-cased_jun24_iter9_7985492_seed-10',
                'job_search': 'neuralmind-bert-base-portuguese-cased_jun24_iter9_7985489_seed-7'
            },
        },
        'MX': {
            'iter0': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_feb27_iter0_3200976_seed-10',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_feb27_iter0_3200974_seed-8',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_feb27_iter0_3200978_seed-12',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_feb27_iter0_3200968_seed-2',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_feb27_iter0_3200967_seed-1'
            },
            'iter1': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_mar12_iter1_3737747_seed-8',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_mar12_iter1_3737745_seed-6',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_mar12_iter1_3737741_seed-2',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_mar12_iter1_3737746_seed-7',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_mar12_iter1_3737745_seed-6'},
            'iter2': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_mar23_iter2_4138955_seed-14',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_mar23_iter2_4138956_seed-15',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_mar23_iter2_4138953_seed-12',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_mar23_iter2_4138951_seed-10',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_mar23_iter2_4138943_seed-2'},
            'iter3': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_mar30_iter3_4375824_seed-2',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_mar30_iter3_4375831_seed-9',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_mar30_iter3_4375832_seed-10',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_mar30_iter3_4375832_seed-10',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_mar30_iter3_4375830_seed-8'},
            'iter4': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_apr2_iter4_4597713_seed-4',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_apr2_iter4_4597718_seed-9',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_apr2_iter4_4597718_seed-9',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_apr2_iter4_4597712_seed-3',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_apr2_iter4_4597710_seed-1'},
            'iter5': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_may17_iter5_6886418_seed-2',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_may17_iter5_6886419_seed-3',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_may17_iter5_6886422_seed-6',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_may17_iter5_6886421_seed-5',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_may17_iter5_6886423_seed-7'
            },
            'iter6': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_may25_iter6_7125251_seed-4',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_may25_iter6_7125254_seed-7',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_may25_iter6_7125255_seed-8',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_may25_iter6_7125252_seed-5',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_may25_iter6_7125251_seed-4'
            },
            'iter7': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_may31_iter7_7272629_seed-7',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_may31_iter7_7272633_seed-11',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_may31_iter7_7272630_seed-8',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_may31_iter7_7272629_seed-7',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_may31_iter7_7272634_seed-12'
            },
            'iter8': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jun5_iter8_7383859_seed-1',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jun5_iter8_7383871_seed-13',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_jun5_iter8_7383867_seed-9',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_jun5_iter8_7383867_seed-9',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_jun5_iter8_7383866_seed-8'
            },
            'iter9': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jun6_iter9_7408605_seed-1',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jun6_iter9_7408615_seed-11',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_jun6_iter9_7408607_seed-3',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_jun6_iter9_7408614_seed-10',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_jun6_iter9_7408609_seed-5'
            },
            'iter10': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jun30_iter10_8222598_seed-12',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jun30_iter10_8222593_seed-7',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_jun30_iter10_8222591_seed-5',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_jun30_iter10_8222587_seed-1',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_jun30_iter10_8222592_seed-6'
            },
            'iter11': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jul20_iter11_8804820_seed-6',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jul20_iter11_8804823_seed-9',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_jul20_iter11_8804817_seed-3',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_jul20_iter11_8804823_seed-9',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_jul20_iter11_8804819_seed-5'
            },
            'iter12': {
                'lost_job_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jul27_iter12_8966158_seed-4',
                'is_hired_1mo': 'dccuchile-bert-base-spanish-wwm-cased_jul27_iter12_8966159_seed-5',
                'is_unemployed': 'dccuchile-bert-base-spanish-wwm-cased_jul27_iter12_8966162_seed-8',
                'job_offer': 'dccuchile-bert-base-spanish-wwm-cased_jul27_iter12_8966155_seed-1',
                'job_search': 'dccuchile-bert-base-spanish-wwm-cased_jul27_iter12_8966167_seed-13'
            },
        }
    }
elif args.method == 1:
    best_model_paths_dict = {
        'US': {
            'iter1': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_apr19_iter1_adaptive_5598877_seed-5',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_apr19_iter1_adaptive_5598877_seed-5',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_apr19_iter1_adaptive_5598883_seed-11',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_apr19_iter1_adaptive_5598880_seed-8',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_apr19_iter1_adaptive_5598880_seed-8'},
            'iter2': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_apr25_iter2_adaptive_5972290_seed-14',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_apr25_iter2_adaptive_5972289_seed-13',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_apr25_iter2_adaptive_5972286_seed-10',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_apr25_iter2_adaptive_5972278_seed-2',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_apr25_iter2_adaptive_5972280_seed-4'
            },
            'iter3': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_apr26_iter3_adaptive_5997887_seed-6',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_apr26_iter3_adaptive_5997886_seed-5',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_apr26_iter3_adaptive_5997886_seed-5',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_apr26_iter3_adaptive_5997890_seed-9',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_apr26_iter3_adaptive_5997893_seed-12'
            },
            'iter4': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_apr27_iter4_adaptive_6026892_seed-10',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_apr27_iter4_adaptive_6026884_seed-2',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_apr27_iter4_adaptive_6026889_seed-7',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_apr27_iter4_adaptive_6026884_seed-2',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_apr27_iter4_adaptive_6026894_seed-12'
            },
            'iter5': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_adaptive_6739858_seed-6',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_adaptive_6739863_seed-11',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_adaptive_6739854_seed-2',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_adaptive_6739861_seed-9',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_adaptive_6739853_seed-1'
            },
            'iter6': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may28_iter6_adaptive_7206891_seed-2',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may28_iter6_adaptive_7206893_seed-4',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may28_iter6_adaptive_7206895_seed-6',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may28_iter6_adaptive_7206894_seed-5',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may28_iter6_adaptive_7206904_seed-15'
            }
        }}

elif args.method == 2:
    best_model_paths_dict = {
        'US': {
            'iter1': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_apr30_iter1_uncertainty_6196561_seed-11',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_apr30_iter1_uncertainty_6196555_seed-5',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_apr30_iter1_uncertainty_6196561_seed-11',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_apr30_iter1_uncertainty_6196560_seed-10',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_apr30_iter1_uncertainty_6196553_seed-3'},
            'iter2': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may1_iter2_uncertainty_6244850_seed-11',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may1_iter2_uncertainty_6244843_seed-4',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may1_iter2_uncertainty_6244841_seed-2',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may1_iter2_uncertainty_6244840_seed-1',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may1_iter2_uncertainty_6244850_seed-11'
            },
            'iter3': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may2_iter3_uncertainty_6314074_seed-4',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may2_iter3_uncertainty_6314072_seed-2',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may2_iter3_uncertainty_6314083_seed-13',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may2_iter3_uncertainty_6314080_seed-10',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may2_iter3_uncertainty_6314071_seed-1'
            },
            'iter4': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may3_iter4_uncertainty_6349919_seed-1',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may3_iter4_uncertainty_6378411_seed-13',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may3_iter4_uncertainty_6378403_seed-5',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may3_iter4_uncertainty_6378407_seed-9',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may3_iter4_uncertainty_6378409_seed-11'
            },
            'iter5': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may11_iter5_uncertainty_6711052_seed-2',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may11_iter5_uncertainty_6711053_seed-3',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may11_iter5_uncertainty_6711059_seed-9',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may11_iter5_uncertainty_6711055_seed-5',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may11_iter5_uncertainty_6711054_seed-4'
            }
        }}

elif args.method == 3:
    best_model_paths_dict = {
        'US': {
            'iter1': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may6_iter1_uncertainty_uncalibrated_6471039_seed-4',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may6_iter1_uncertainty_uncalibrated_6471042_seed-7',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may6_iter1_uncertainty_uncalibrated_6471047_seed-12',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may6_iter1_uncertainty_uncalibrated_6471036_seed-1',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may6_iter1_uncertainty_uncalibrated_6471048_seed-13'},
            'iter2': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may7_iter2_uncertainty_uncalibrated_6518196_seed-10',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may7_iter2_uncertainty_uncalibrated_6518200_seed-14',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may7_iter2_uncertainty_uncalibrated_6518187_seed-1',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may7_iter2_uncertainty_uncalibrated_6518197_seed-11',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may7_iter2_uncertainty_uncalibrated_6518187_seed-1'
            },
            'iter3': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may8_iter3_uncertainty_uncalibrated_6583469_seed-5',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may8_iter3_uncertainty_uncalibrated_6583465_seed-1',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may8_iter3_uncertainty_uncalibrated_6583472_seed-8',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may8_iter3_uncertainty_uncalibrated_6583478_seed-14',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may8_iter3_uncertainty_uncalibrated_6583472_seed-8'
            },
            'iter4': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may10_iter4_uncertainty_uncalibrated_6653463_seed-2',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may10_iter4_uncertainty_uncalibrated_6653473_seed-12',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may10_iter4_uncertainty_uncalibrated_6653473_seed-12',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may10_iter4_uncertainty_uncalibrated_6653464_seed-3',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may10_iter4_uncertainty_uncalibrated_6653472_seed-11'
            },
            'iter5': {
                'lost_job_1mo': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_uncertainty_uncalibrated_6737085_seed-12',
                'is_hired_1mo': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_uncertainty_uncalibrated_6737082_seed-9',
                'is_unemployed': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_uncertainty_uncalibrated_6737074_seed-1',
                'job_offer': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_uncertainty_uncalibrated_6737077_seed-4',
                'job_search': 'DeepPavlov-bert-base-cased-conversational_may12_iter5_uncertainty_uncalibrated_6737086_seed-13'
            }
        }}

for label in ["lost_job_1mo", "is_unemployed", "job_search", "is_hired_1mo", "job_offer"]:
    logger.info(f'*****************************{label}*****************************')
    model_path = os.path.join('/scratch/mt4493/twitter_labor/trained_models', args.country_code,
    best_model_paths_dict[args.country_code][f'iter{str(args.iteration_number)}'][label],
    label, 'models', 'best_model')
    onnx_path = os.path.join(model_path, 'onnx')

    try:
        shutil.rmtree(onnx_path)  # deleting onxx folder and contents, if exists, conversion excepts
    except:
        logger.info('no existing folder, creating one')
        os.makedirs(onnx_path)

    logger.info('>> converting..')
    convert(framework="pt",
            model=model_path,
            tokenizer=convert_model_path_to_model_name(model_path),
            output=Path(os.path.join(onnx_path, 'converted.onnx')),
            opset=11,
            pipeline_name='sentiment-analysis')

    logger.info('>> ONNX optimization')
    optimized_output = optimize(Path(os.path.join(onnx_path, 'converted.onnx')))
    logger.info('>> Quantization')
    quantized_output = quantize(optimized_output)

    logger.info('>> Verification')
    verify(Path(os.path.join(onnx_path, 'converted.onnx')))
    verify(optimized_output)
    verify(quantized_output)
