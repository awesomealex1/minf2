import torch
import os

class MetricsLogger():

    def __init__(self, exp_name, dataset, model, seed):
        self.dir_path = f'experiment_results/{dataset}_{model}/{seed}'
        self.log_file_name = "metrics.txt"
        self.final_log_file_name = "final_metrics.txt"
        self.hyperparam_file_name = "hp_results.txt"
        self.args_file_name = "args.txt"

        os.makedirs(self.dir_path, exist_ok=True)
        with open(os.path.join(self.dir_path, self.log_file_name), "w") as f:
            pass

    def log_epoch_acc(self, epoch, train_acc, val_acc, test_acc):
        with open(os.path.join(self.dir_path, self.log_file_name), "a") as f:
            f.write(f"Epoch:{epoch+1} Train_accuracy:{train_acc} Val_accuracy:{val_acc} Test_accuracy:{test_acc}\n")

    def log_all_epochs_accs(self, epochs, train_accs, val_accs, test_accs):
        with open(os.path.join(self.dir_path, self.final_log_file_name), 'w') as f:
            for i in range(epochs):
                f.write(f"Epoch:{i+1} Train_accuracy:{train_accs[i]} Val_accuracy:{val_accs[-1]} Test_accuracy:{test_accs[i]}\n")

    def save_model(self, model):
        torch.save(model.state_dict(), f'{self.dir_path}/weights.pt')

    def save_final_model(self, model):
        torch.save(model.state_dict(), f'{self.dir_path}/final_weights.pt')
    
    def save_deltas(self, deltas):
        torch.save(deltas, f'{self.dir_path}/deltas.pt')
    
    def save_final_deltas(self, deltas):
        torch.save(deltas, f'{self.dir_path}/final_deltas.pt')

    def read_final_deltas(self):
        deltas = torch.load(f'{self.dir_path}/final_deltas.pt')
        return deltas
    
    def log_hyperparam_result(self, best_params, best_value):
        with open(os.path.join(self.dir_path, self.hyperparam_file_name), "a") as f:
            f.write(f"Best_params: {best_params} Best_value: {best_value}\n")
    
    def log_args(self, args):
        with open(os.path.join(self.dir_path, self.args_file_name), "a") as f:
            for arg in args.keys():
                f.write(f"{args}: {args[arg]}\n")

    def save_spectrum(self, spectrum):
        with open(os.path.join(self.dir_path, self.log_file_name), "a") as f:
            f.write(f"Spectrum: {spectrum}")