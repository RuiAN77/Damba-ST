import torch
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
from tqdm import tqdm
from lib.data_process import get_key_from_value


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_dataloader, val_dataloader, test_dataloader, scaler_dict,
                 args, scheduler):
        super(Trainer, self).__init__()
        self.model = model
        self.args = args
        self.loss = loss
        self.optimizer = optimizer
        self.num_nodes_dict = args.num_nodes_dict
        self.client_no = args.client_no
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.scaler_dict = scaler_dict
        self.scheduler = scheduler
        self.batch_seen = 0
        self.best_path = os.path.join(self.args.log_dir, self.args.save_pretrain_path)
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        # log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

    def send_models(self):
        for client in self.model:
            start_time = time.time()
            for (nn, np), (on, op) in zip(self.model[0].named_parameters(), client.named_parameters()):
                if 'layer' not in nn:
                    op.data = np.data.clone()
    def multi_train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        # train_loss_list = []
        val_loss_list = []
        for i in range(self.args.global_rounds + 1):
            self.send_models()
            for epoch in tqdm(range(self.args.epochs)):
                # Train
                # start_time = time.time()
                train_epoch_loss = self.multi_train_eps()
                # training_time = time.time() - start_time

                if train_epoch_loss > 1e6:
                    self.logger.warning('Gradient explosion detected. Ending...')
                    break

                if self.args.mode != 'pretrain' and self.args.val_ratio > 0:
                    # Val
                    val_epoch_loss = self.multi_val_epoch(epoch)
                    # Best state and early stop epoch
                    val_loss_list.append(val_epoch_loss)
                    if val_epoch_loss < best_loss:
                        best_loss = val_epoch_loss
                        not_improved_count = 0
                        best_state = True
                    else:
                        not_improved_count += 1
                        best_state = False

                    # early stop
                    if self.args.early_stop:
                        if not_improved_count == self.args.early_stop_patience:
                            self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                             "Training stops.".format(self.args.early_stop_patience))
                            break

                    # save the best state
                    if best_state == True:
                        self.logger.info('Current best model saved!')
                        # self.test(self.model, self.args, self.scaler_dict, self.test_dataloader, self.logger)
                        best_model = copy.deepcopy(self.model.state_dict())
            self.receive_models()
            self.aggregate_parameters()
        # test
        if self.args.mode != 'pretrain' and self.args.val_ratio > 0:
            self.model.load_state_dict(best_model)
            self.test(self.model, self.args, self.scaler_dict, self.test_dataloader, self.logger)
        print("Pre-train finish.")

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    # Personalized for Lora (Global) others local
    def set_parameters(self, client, model):
        for (nn, np), (on, op) in zip(model.named_parameters(), client.named_parameters()):
            if 'layer' not in nn:
                op.data = np.data.clone()

    def multi_train_eps(self):
        self.model[0].train()
        total_loss = 0
        step = 0

        for inputs, targets in self.train_dataloader:
            inputs, targets = inputs.squeeze(0).to(self.args.device), targets.squeeze(0).to(self.args.device)
            select_dataset = get_key_from_value(self.num_nodes_dict, inputs.shape[2])
            model_no=self.client_no[select_dataset]
            print("now select_dataset and client",select_dataset, model_no)
            out = self.model[model_no](inputs, targets, select_dataset, batch_seen=None)
            self.optimizer.zero_grad()
            loss_pred = self.loss(out, targets[..., :self.args.output_dim], self.scaler_dict[select_dataset])
            loss = loss_pred
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model[model_no].parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            # learning rate decay
            if self.args.lr_decay:
                self.scheduler.step()
            total_loss += loss.item()
            step += 1
            if step % self.args.log_step == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    "step:  " + str(step) + "  train loss is:  " + str(
                        total_loss / step) + "  current_lr is:  " + str(
                        current_lr))
            if step % self.args.save_step == 0 and self.args.debug:
                best_model = copy.deepcopy(self.model[model_no].state_dict())
                torch.save(best_model, self.best_path)
                self.logger.info("Saving current best model to " + self.best_path)
        train_loss = total_loss / len(self.train_dataloader)
        return train_loss

    def multi_val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in self.val_dataloader:
                inputs, targets = inputs.squeeze(0).to(self.args.device), targets.squeeze(0).to(self.args.device)
                select_dataset = get_key_from_value(self.num_nodes_dict, inputs.shape[2])
                out = self.model(inputs, targets, select_dataset, batch_seen=None)
                loss_pred = self.loss(out, targets[..., 0:self.args.output_dim], self.scaler_dict[select_dataset])
                if not torch.isnan(loss_pred):
                    total_val_loss += loss_pred.item()
                val_loss = total_val_loss / len(self.val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    @staticmethod
    def test(model, args, scaler_dict, test_dataloader, logger, path=None):
        if path != None:
            if torch.cuda.device_count() > 1:
                model.load_state_dict(torch.load(path))
            else:
                model_weights = {k.replace('module.', ''): v for k, v in torch.load(path).items()}
                model.load_state_dict(model_weights)
            model.to(args.device)
        model.eval()

        with torch.no_grad():
            mae = 0
            rmse = 0
            mape = 0
            total_count = 0
            total_mape_count = 0
            total_batch = 0
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.squeeze(0).to(args.device), targets.squeeze(0).to(args.device)
                select_dataset = get_key_from_value(args.num_nodes_dict, inputs.shape[2])
                output = model(inputs, targets, select_dataset, batch_seen=None)
                if args.real_value == False:
                    output = scaler_dict[select_dataset].inverse_transform(output)
                    y_lbl = scaler_dict[select_dataset].inverse_transform(targets[..., :args.output_dim])
                else:
                    y_lbl = targets[..., :args.output_dim]
                batch_mae, batch_rmse, batch_mape, batch_mse, corr, mae_count, rmse_count, mse_count, mape_count = \
                    All_Metrics(output, y_lbl, args.mae_thresh, args.mape_thresh)
                mae += batch_mae * mae_count
                rmse += batch_mse * rmse_count
                mape += batch_mape * mape_count
                total_count += mae_count
                total_mape_count += mape_count
                total_batch += len(y_lbl)
                if args.model == 'Damba':
                    print(total_batch, batch_mae, batch_rmse, batch_mape, total_count, total_mape_count)
        mae /= total_count
        rmse = (rmse / total_count) ** 0.5
        mape /= total_mape_count
        print('last batch', output.shape, y_lbl.shape)
        print(total_batch, total_count, total_mape_count)

        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, CORR:{:.4f}".format(
            mae, rmse, mape * 100, corr))
