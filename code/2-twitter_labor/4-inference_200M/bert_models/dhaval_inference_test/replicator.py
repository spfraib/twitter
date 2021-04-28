import os

for model_type in ['converted.onnx', 'converted-optimized.onnx', 'converted-optimized-quantized.onnx']:
    for batchsize in [1, 2, 5, 10, 20]:
    # for batchsize in [1, 2, 5, 10, 20, 50, 100]:
        for replication in range(5):
            os.system(
                'python inference_ONNX_bert_100M_random_dhaval_optimized_batch_compared_torch_and_onnx.py --input_path '
                '/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/US/test --output_path '
                '/Users/dval/work_temp/twitter_from_nyu/output_faster --country_code US --iteration_number 2 '
                '--batchsize {} '
                '--model_type {} --replication {}'.format(
                    batchsize,
                    model_type,
                    replication
                )
            )
            # break
        # break
    # break


