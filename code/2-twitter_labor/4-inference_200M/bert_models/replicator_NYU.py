import os
import onnxruntime as ort
print(ort.get_device())

for model_type in ['converted.onnx', 'converted-optimized.onnx', 'converted-optimized-quantized.onnx']:
    for batchsize in [1, 2, 5, 10, 20]:
    # for batchsize in [1, 2, 5, 10, 20, 50, 100]:
        for replication in range(5):
            print(model_type, batchsize, replication)
            os.system(
                'python inference_ONNX_bert_100M_random_dhaval_optimized_batch_compared_torch_and_onnx_NYU.py --input_path '
                '/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/US/test --output_path '
                '/scratch/mt4493/twitter_labor/code/twitter/code/2-twitter_labor/4-inference_200M/bert_models/temp_output_speedtest --country_code US --iteration_number 2 '
                '--batchsize {} '
                '--model_type {} --replication {}'.format(
                    batchsize,
                    model_type,
                    replication
                )
            )
            break
        break
    break



