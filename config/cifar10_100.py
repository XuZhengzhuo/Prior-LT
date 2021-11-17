# model
arch = 'resnet32'

# dataset
dataset = 'cifar10'  # or 'cifar100'
imb_type = 'exp'  # or 'step'
num_classes = int(dataset[5:])
imb_factor = 0.01
train_cls_num_list = None
inf_label_distribution = None

if dataset == 'cifar10':
    h_class_idx = [0, 3]
    m_class_idx = [3, 7]
    t_class_idx = [7, 10]
else:
    h_class_idx = [0, 33]
    m_class_idx = [33, 66]
    t_class_idx = [66, 100]

# load setting
workers = 4
seed = 0
rand_number = 0

# gpu
gpu = 0

# train setting
epochs = 200
batch_size = 64  # will double if mix
lr = 0.1
start_epoch = 0
momentum = 0.9
weight_decay = 2e-4

# mixup manners
mix_type = 'unimix'
mix_stop_epoch = 200

# alp=1. and tau=0. equals to origin mixup
unimix_alp = 0.8
unimix_tau = -0.5

# loss
loss_type = 'Bayias'  # or 'CE'

# checkpoint
resume = ''  # relative path to ckpt
save_ckpt_epoch = mix_stop_epoch  # save ckpt for finetune

# debug info
cfg_name = 'Final'  # the main store path
debug = False  # mkdir or not
note = f'bs_64_alp_0.8_tau_0.5'  # for better visualization