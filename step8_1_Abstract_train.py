# version: 0110

# Constant
GPU_ID = "1"
SEED = 42     # random seed
Gflag = "True" # "True" or "False"


step_flag = "train_hyper_bestTest"
task_flag = step_flag + "_EGCN"


date_flag = "0706"
hyper_flag = step_flag + "_output"


# max_aa_atom = 630
max_pocket_len = 54
max_atom_num = 48
max_bond_num = 53
# Package
import fixedLoss_ThreeTask
from fixedLoss_ThreeTask import MultitaskMnistLoss
from step7_1_model import Fingerprint, Masked_BCELoss, Masked_MSELoss

# Path_input
prex_input_path = "../"
prex_path = "./data_process/"
data_path_dict = {
    "model_dataset": prex_path + "step3_nonCovalent/out3_2_model_dataset_0607.pickle",
    "nonCovalent_dict": prex_path + "step3_nonCovalent/out3_1_interaction_dict_0607.pickle",
}




