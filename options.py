import argparse

class training_options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--raw_data_folder', type=str, default='', help='path to folder includes all .nii.gz files of this dataset')
        self.parser.add_argument('--site', type=str, default='Center_01', help='in this dataset, which site for training. Also part of the name of .nii.gz files')
        self.parser.add_argument('--model', type=str, default='UNet', help='model used for training')
        self.parser.add_argument('--exp_name', type=str,required=True, help='model and results are saved in checkpoint_dir/exp_name/')
        self.parser.add_argument('--checkpoint_dir', type=str, default='/home/checkpoints', help='absolute path to save the model and results')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--seed', type=int, default=10, help='fix the random seed')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
        self.parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        self.parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
        self.parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train for')
        self.parser.add_argument('--model_save_freq', default=10, type=int, help='save model every model_save_freq epochs')
        self.parser.add_argument('--test_freq', default=1, type=int, help='test model every test_freq epochs')
        self.parser.add_argument('--loss_param_alpha', type=float, default=1, help='alpha for fixed loss, multiplied by focal')
        self.parser.add_argument('--loss_param_beta', type=float, default=1, help='beta for fixed loss, multiplied by dice')
        self.parser.add_argument('--early_stopping', type=int, default=200, help='early stopping epoch')
        self.parser.add_argument('--task', type=str, default=None, help='which task to run: WMH')
        self.parser.add_argument('--single_axis',  action='store_true',help='Set True for single axis slicing')
        self.parser.add_argument('--augment', action='store_true', help='Set True for training slice augment slicing')
        self.parser.add_argument('--test_site', nargs='+', default='trainingmiccai21', help='for SsDG, define which sites for testing')
        self.parser.add_argument('--eval', default=False ,action='store_true',help='Set eval mode for testing')
        self.parser.add_argument('--continue_training', default=False ,action='store_true',help='If true, continue training from checkpoint')
        self.parser.add_argument('--continue_checkpoint', type=str, help='path to checkpoint to continue training from')
        self.parser.add_argument('--training_slice_path', type=str, default='', help='The folder where training slices are saved')

    def parse(self):
        if not self.initialized:
            self.initialize()
            self.opt = self.parser.parse_args()

            return self.opt
