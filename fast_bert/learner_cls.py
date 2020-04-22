import os
from .data_cls import BertDataBunch, InputExample, InputFeatures
from .learner_util import Learner
from torch import nn
from typing import List
import torch.cuda as cutorch


from .modeling import (
    BertForMultiLabelSequenceClassification,
    XLNetForMultiLabelSequenceClassification,
    RobertaForMultiLabelSequenceClassification,
    DistilBertForMultiLabelSequenceClassification,
    CamembertForMultiLabelSequenceClassification,
    AlbertForMultiLabelSequenceClassification,
)

from .bert_layers import BertLayerNorm
from fastprogress.fastprogress import master_bar, progress_bar
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

import json
import time

from pathlib import Path

from torch.optim.lr_scheduler import _LRScheduler, Optimizer

from tensorboardX import SummaryWriter

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    CamembertConfig,
    CamembertForSequenceClassification,
    CamembertTokenizer,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)


MODEL_CLASSES = {
    "bert": (
        BertConfig,
        (BertForSequenceClassification, BertForMultiLabelSequenceClassification),
        BertTokenizer,
    ),
    "xlnet": (
        XLNetConfig,
        (XLNetForSequenceClassification, XLNetForMultiLabelSequenceClassification),
        XLNetTokenizer,
    ),
    "xlm": (
        XLMConfig,
        (XLMForSequenceClassification, XLMForSequenceClassification),
        XLMTokenizer,
    ),
    "roberta": (
        RobertaConfig,
        (RobertaForSequenceClassification, RobertaForMultiLabelSequenceClassification),
        RobertaTokenizer,
    ),
    "distilbert": (
        DistilBertConfig,
        (
            DistilBertForSequenceClassification,
            DistilBertForMultiLabelSequenceClassification,
        ),
        DistilBertTokenizer,
    ),
    "albert": (
        AlbertConfig,
        (AlbertForSequenceClassification, AlbertForMultiLabelSequenceClassification),
        AlbertTokenizer,
    ),
    "camembert": (
        CamembertConfig,
        (
            CamembertForSequenceClassification,
            CamembertForMultiLabelSequenceClassification,
        ),
        CamembertTokenizer,
    ),
}


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm
except Exception:
    from .bert_layers import BertLayerNorm as FusedLayerNorm


class BertLearner(Learner):
    @staticmethod
    def from_pretrained_model(
        dataBunch,
        pretrained_path,
        output_dir,
        metrics,
        device,
        logger,
        finetuned_wgts_path=None,
        multi_gpu=True,
        is_fp16=True,
        loss_scale=0,
        warmup_steps=0,
        fp16_opt_level="O1",
        grad_accumulation_steps=1,
        multi_label=False,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        logging_steps=100,
    ):

        model_state_dict = None

        model_type = dataBunch.model_type

        config_class, model_class, _ = MODEL_CLASSES[model_type]

        config = config_class.from_pretrained(
            str(pretrained_path), num_labels=len(dataBunch.labels)
        )

        if finetuned_wgts_path:
            model_state_dict = torch.load(finetuned_wgts_path)
        else:
            model_state_dict = None

        if multi_label is True:
            print(str(pretrained_path))
            print(type(str(pretrained_path)))
            model = model_class[1].from_pretrained(
                str(pretrained_path), config=config, state_dict=model_state_dict
            )
        else:
            model = model_class[0].from_pretrained(
                str(pretrained_path), config=config, state_dict=model_state_dict
            )

        model.to(device)
#         print(model)

        return BertLearner(
            dataBunch,
            model,
            str(pretrained_path),
            output_dir,
            metrics,
            device,
            logger,
            multi_gpu,
            is_fp16,
            loss_scale,
            warmup_steps,
            fp16_opt_level,
            grad_accumulation_steps,
            multi_label,
            max_grad_norm,
            adam_epsilon,
            logging_steps,
        )

    def __init__(
        self,
        data: BertDataBunch,
        model: nn.Module,
        pretrained_model_path,
        output_dir,
        metrics,
        device,
        logger,
        multi_gpu=True,
        is_fp16=True,
        loss_scale=0,
        warmup_steps=0,
        fp16_opt_level="O1",
        grad_accumulation_steps=1,
        multi_label=False,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        logging_steps=100,
        # training_results = output_dir+"/results.csv",
    ):

        super(BertLearner, self).__init__(
            data,
            model,
            pretrained_model_path,
            output_dir,
            device,
            logger,
            multi_gpu,
            is_fp16,
            warmup_steps,
            fp16_opt_level,
            grad_accumulation_steps,
            max_grad_norm,
            adam_epsilon,
            logging_steps,
            
        )

        # Classification specific attributes
        self.multi_label = multi_label
        self.metrics = metrics
        self.results_all_epochs = {}

    def save_results(self):

        path_to_file = self.output_dir / "results.json"

        print('>> saving to.. ', path_to_file)
        with open(path_to_file, 'w+') as json_file: #w+ for delete the original content then read/write if file exists, otherwise create the file
            json.dump(self.results_all_epochs, json_file)


    ### Train the model ###
    def fit(
        self,
        epochs,
        lr,
        validate=True,
        schedule_type="warmup_cosine",
        optimizer_type="lamb",
    ):

        tensorboard_dir = self.output_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)

        # Train the model
        tb_writer = SummaryWriter(tensorboard_dir)

        train_dataloader = self.data.train_dl
        if self.max_steps > 0:
            t_total = self.max_steps
            self.epochs = (
                self.max_steps // len(train_dataloader) // self.grad_accumulation_steps
                + 1
            )
        else:
            t_total = len(train_dataloader) // self.grad_accumulation_steps * epochs

        # Prepare optimiser
        optimizer = self.get_optimizer(lr, optimizer_type=optimizer_type)
        # input('hey')

        # get the base model if its already wrapped around DataParallel
        if hasattr(self.model, "module"):
            # input('module')
            self.model = self.model.module

        if self.is_fp16:
            # print('fp16')
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex to use fp16 training")
            self.model, optimizer = amp.initialize(
                self.model, optimizer, opt_level=self.fp16_opt_level
            )

        # print(type(self.model))
        # input('hey')
        # Get scheduler
        scheduler = self.get_scheduler(
            optimizer, t_total=t_total, schedule_type=schedule_type
        )

        # Parallelize the model architecture
        if self.multi_gpu is True:
            # input('multi_gpu')
            self.model = torch.nn.DataParallel(self.model)

        # Start Training
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_dataloader.dataset))
        self.logger.info("  Num Epochs = %d", epochs)
        self.logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.data.train_batch_size * self.grad_accumulation_steps,
        )
        self.logger.info(
            "  Gradient Accumulation steps = %d", self.grad_accumulation_steps
        )
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epoch_step = 0
        tr_loss, logging_loss, epoch_loss = 0.0, 0.0, 0.0
        self.model.zero_grad()
        pbar = master_bar(range(epochs))

        for epoch in pbar:

            self.save_model(epoch=epoch)
            start_time = time.time()

            epoch_step = 0
            epoch_loss = 0.0
            for step, batch in enumerate(progress_bar(train_dataloader, parent=pbar)):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }

                if self.model_type in ["bert", "xlnet"]:
                    inputs["token_type_ids"] = batch[2]

                outputs = self.model(**inputs)

                # print('>>>>>', outputs)

                loss = outputs[
                    0
                ]  # model outputs are always tuple in pytorch-transformers (see doc)

                if self.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training
                if self.grad_accumulation_steps > 1:
                    loss = loss / self.grad_accumulation_steps

                if self.is_fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), self.max_grad_norm
                    )
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                tr_loss += loss.item()
                epoch_loss += loss.item()
                if (step + 1) % self.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()

                    self.model.zero_grad()
                    global_step += 1
                    epoch_step += 1

                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        if validate:
                            # evaluate model
                            results = self.validate()
                            for key, value in results.items():
                                tb_writer.add_scalar(
                                    "eval_{}".format(key), value, global_step
                                )
                                self.logger.info(
                                    "eval_{} after step {}: {}: ".format(
                                        key, global_step, value
                                    )
                                )

                        # Log metrics
                        self.logger.info(
                            "lr after step {}: {}".format(
                                global_step, scheduler.get_lr()[0]
                            )
                        )
                        self.logger.info(
                            "train_loss after step {}: {}".format(
                                global_step,
                                (tr_loss - logging_loss) / self.logging_steps,
                            )
                        )
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / self.logging_steps,
                            global_step,
                        )

                        logging_loss = tr_loss

            # Evaluate the model after every epoch
            if validate:
                results = self.validate()
                for key, value in results.items():
                    self.logger.info(
                        "eval_{} after epoch {}: {}: ".format(key, (epoch + 1), value)
                    )
                results['epoch'] = epoch
                results['train_loss'] = epoch_loss
                results['lr'] = scheduler.get_lr()[0]
                results['time'] = str(time.time() - start_time)

                self.results_all_epochs[epoch] = results
                self.save_results()

            # Log metrics
            self.logger.info(
                "lr after epoch {}: {}".format((epoch + 1), scheduler.get_lr()[0])
            )
            self.logger.info(
                "train_loss after epoch {}: {}".format(
                    (epoch + 1), epoch_loss / epoch_step
                )
            )
            self.logger.info("\n")

        tb_writer.close()
        return global_step, tr_loss / global_step

    ### Evaluate the model
    def validate(self):
        self.logger.info("Running evaluation")

        self.logger.info("  Num examples = %d", len(self.data.val_dl.dataset))
        self.logger.info("  Batch size = %d", self.data.val_batch_size)

        all_logits = None
        all_labels = None

        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0

        preds = None
        out_label_ids = None

        validation_scores = {metric["name"]: 0.0 for metric in self.metrics}

        for step, batch in enumerate(progress_bar(self.data.val_dl)):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }

                if self.model_type in ["bert", "xlnet"]:
                    inputs["token_type_ids"] = batch[2]

                outputs = self.model(**inputs)
                # print(outputs)
#                 print(outputs.shape)
                
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            nb_eval_examples += inputs["input_ids"].size(0)

            if all_logits is None:
                all_logits = logits
            else:
                all_logits = torch.cat((all_logits, logits), 0)

            if all_labels is None:
                all_labels = inputs["labels"]
            else:
                all_labels = torch.cat((all_labels, inputs["labels"]), 0)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps

        # Evaluation metrics
        for metric in self.metrics:
            validation_scores[metric["name"]] = metric["function"](
                all_logits, all_labels
            )

        results = {"loss": eval_loss}
        results.update(validation_scores)

        return results


    ### Return Predictions ###
    def predict_batch(self, texts=None):

        if texts:
            dl = self.data.get_dl_from_texts(texts)
        elif self.data.test_dl:
            dl = self.data.test_dl
        else:
            dl = self.data.val_dl

        all_logits = None

        self.model.eval()
        for step, batch in enumerate(dl):
            batch = tuple(t.to(self.device) for t in batch)
            
#             print('batch size', batch[0].shape)
#             t = torch.cuda.get_device_properties(0).total_memory
#             c = torch.cuda.memory_cached(0)
#             a = torch.cuda.memory_allocated(0)
#             f = c-a  # free inside cache
#             print(f/1024/1024, 'gb free', t/1024/1024, 'gb total')

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}

            if self.model_type in ["bert", "xlnet"]:
                inputs["token_type_ids"] = batch[2]

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs[0]
                if self.multi_label:
                    logits = logits.sigmoid()
                # elif len(self.data.labels) == 2:
                #     logits = logits.sigmoid()
                else:
                    logits = logits.softmax(dim=1)

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate(
                    (all_logits, logits.detach().cpu().numpy()), axis=0
                )
        print('inference done')

        result_df = pd.DataFrame(all_logits, columns=self.data.labels)
#         print(result_df.head())
#         results = result_df.to_dict("record")
#         print(results)

#         return all_logits
        return result_df
#         return [sorted(x.items(), key=lambda kv: kv[1], reverse=True) for x in results] #for some reason, this is sometime not sorted correctly
