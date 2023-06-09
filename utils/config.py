from easydict import EasyDict

CONFIG = EasyDict({})
CONFIG.is_default = True
CONFIG.version = "baseline"
CONFIG.debug = False
# choices from train,evaluate,inference
CONFIG.phase = "train"
# distributed training
CONFIG.dist = False
# global variables which will be assigned in the runtime
CONFIG.local_rank = 0
CONFIG.gpu = 0
CONFIG.world_size = 1
CONFIG.devices = (0,)


# ===============================================================================
# Model config
# ===============================================================================
CONFIG.model = EasyDict({})
CONFIG.model.arch = 'SparseMat'

# Model -> Architecture config
CONFIG.model.in_channel = 3
CONFIG.model.hr_channel = 32
# global modules (ppm, aspp)
CONFIG.model.global_module = "ppm"
CONFIG.model.pool_scales = (1,2,3,6)
CONFIG.model.ppm_channel = 256
CONFIG.model.atrous_rates = (12, 24, 36)
CONFIG.model.aspp_channel = 256
CONFIG.model.with_norm = True
CONFIG.model.with_aspp = True
CONFIG.model.dilation_kernel = 15
CONFIG.model.max_n_pixel = 4000000

# ===============================================================================
# Dataloader config
# ===============================================================================

CONFIG.aug = EasyDict({})
CONFIG.aug.rescale_size = 320
CONFIG.aug.crop_size = 288
CONFIG.aug.patch_crop_size = (320,640)
CONFIG.aug.patch_load_size = 320

CONFIG.data = EasyDict({})
CONFIG.data.workers = 0
CONFIG.data.dataset = None
CONFIG.data.composite = False
CONFIG.data.filelist = None
CONFIG.data.filelist_train = None
CONFIG.data.filelist_val = None
CONFIG.data.filelist_test = None


# ===============================================================================
# Loss config
# ===============================================================================
CONFIG.loss = EasyDict({})
CONFIG.loss.alpha_loss_weights = [1.0,1.0,1.0,1.0]
CONFIG.loss.with_composition_loss = False
CONFIG.loss.composition_loss_weight = 1.0

# ===============================================================================
# Training config
# ===============================================================================
CONFIG.train = EasyDict({})

CONFIG.train.num_workers = 4
CONFIG.train.batch_size = 8
# epochs
CONFIG.train.start_epoch = 0
CONFIG.train.epoch = 100
CONFIG.train.epoch_decay = 95
# basic learning rate of optimizer
CONFIG.train.lr = 1e-5
CONFIG.train.min_lr = 1e-8
CONFIG.train.reset_lr = False
CONFIG.train.adaptive_lr = False
# beta1 and beta2 for Adam
CONFIG.train.optim = "Adam"
CONFIG.train.eps = 1e-5
CONFIG.train.beta1 = 0.9
CONFIG.train.beta2 = 0.999
CONFIG.train.momentum = 0.9
CONFIG.train.weight_decay = 1e-4
# clip large gradient
CONFIG.train.clip_grad = True
# reset the learning rate (this option will reset the optimizer and learning rate scheduler and ignore warmup)
CONFIG.train.pretrained_model = None
CONFIG.train.resume = False

CONFIG.train.rescale_size = 320
CONFIG.train.crop_size = 288
CONFIG.train.color_space = 3


# ===============================================================================
# Testing config
# ===============================================================================
CONFIG.test = EasyDict({})
# test image scale to evaluate, "origin" or "resize" or "crop"
CONFIG.test.num_workers = 4
CONFIG.test.batch_size = 1
CONFIG.test.rescale_size = 320
CONFIG.test.max_size = 1920
CONFIG.test.patch_size = 320
CONFIG.test.checkpoint = None
CONFIG.test.save = False
CONFIG.test.save_dir = None
CONFIG.test.cascade = False
# "best_model" or "latest_model" or other base name of the pth file.


# ===============================================================================
# Logging config
# ===============================================================================
CONFIG.log = EasyDict({})
CONFIG.log.log_dir = None
CONFIG.log.viz_dir = None
CONFIG.log.save_frq = 2000
CONFIG.log.print_frq = 20
CONFIG.log.test_frq = 1
CONFIG.log.viz = True
CONFIG.log.show_all = True


# ===============================================================================
# util functions
# ===============================================================================
def parse_config(custom_config, default_config=CONFIG, prefix="CONFIG"):
    """
    This function will recursively overwrite the default config by a custom config
    :param default_config:
    :param custom_config: parsed from config/config.toml
    :param prefix: prefix for config key
    :return: None
    """
    if "is_default" in default_config:
        default_config.is_default = False

    for key in custom_config.keys():
        full_key = ".".join([prefix, key])
        if key not in default_config:
            raise NotImplementedError("Unknown config key: {}".format(full_key))
        elif isinstance(custom_config[key], dict):
            if isinstance(default_config[key], dict):
                parse_config(default_config=default_config[key],
                            custom_config=custom_config[key],
                            prefix=full_key)
            else:
                raise ValueError("{}: Expected {}, got dict instead.".format(full_key, type(custom_config[key])))
        else:
            if isinstance(default_config[key], dict):
                raise ValueError("{}: Expected dict, got {} instead.".format(full_key, type(custom_config[key])))
            else:
                default_config[key] = custom_config[key]


def load_config(config_path):
    import toml
    with open(config_path) as fp:
        custom_config = EasyDict(toml.load(fp))
    parse_config(custom_config=custom_config)
    return CONFIG


if __name__ == "__main__":
    from pprint import pprint

    pprint(CONFIG)
    load_config("../config/example.toml")
    pprint(CONFIG)
