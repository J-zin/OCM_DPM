from core.diffusion.schedule import NamedSchedule
import configs.default as default
from .models import *


def get_train_config(**hparams):
    hparams.setdefault('schedule', NamedSchedule('cosine', 4000))
    hparams.setdefault('shift1', True)  # follow original Improved DDPM
    hparams['N'] = hparams['schedule'].N

    config = default.get_train_config(**hparams)
    config.models = ml_collections.ConfigDict()
    config.dataset = default.get_imagenet64_config(**hparams)
    if hparams['method'] == 'pred_eps':
        config.models.model = get_iddpm_unet_out3_config(**hparams)
        config.criterion = default.get_dt_dsm_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps', **hparams)
    elif hparams['method'] == 'pred_eps_hes_pretrained':
        hparams['rev_var_type'] = 'optimal'
        config.models.model = get_iddpm_unet_double_pretrained_config(**hparams)
        config.criterion = default.get_dt_dsdm_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_hes', **hparams)
    elif hparams['method'] == 'pred_eps_eps2_pretrained':
        hparams['rev_var_type'] = 'optimal'
        config.models.model = get_iddpm_unet_double_pretrained_config(**hparams)
        config.criterion = default.get_dt_dsdm_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_eps2', **hparams)
    elif hparams['method'] == 'pred_eps_epsc_pretrained':
        hparams['rev_var_type'] = 'optimal'
        config.models.model = get_iddpm_unet_double_pretrained_config(**hparams)
        config.criterion = default.get_dt_dsdm_err_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_epsc', **hparams)
    else:
        raise NotImplementedError

    config.evaluator = default.get_train_evaluator_config(**hparams)
    return config
