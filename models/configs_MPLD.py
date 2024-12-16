

import ml_collections
'''
********************************************************
                   Swin Transformer
********************************************************
if_transskip (bool): Enable skip connections from Transformer Blocks
if_convskip (bool): Enable skip connections from Convolutional Blocks
patch_size (int | tuple(int)): Patch size. Default: 4
in_chans (int): Number of input image channels. Default: 2 (for moving and fixed images)

'''
def get_3DMPLD_config():
    '''
    Trainable params: 15,201,579
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (32, 256, 384)
    return config

def get_3DMPLDNoRelativePosEmbd_config():
    '''
    Trainable params: 15,201,579
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (32, 256, 384)
    return config

def get_3DMPLDSin_config():
    '''

    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = True
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (32, 256, 384)
    config.pos_embed_method = 'relative'
    return config

def get_3DMPLDLrn_config():
    '''

    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = True
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (32, 256, 384)
    return config

def get_3DMPLDNoConvSkip_config():
    '''

    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = False
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (32, 256, 384)
    config.pos_embed_method = 'relative'
    return config

def get_3DMPLDNoTransSkip_config():
    '''

    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = False
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (32, 256, 384)
    return config

def get_3DMPLDNoSkip_config():
    '''

    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = False
    config.if_convskip = False
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (32, 256, 384)
    return config

def get_3DMPLDLarge_config():
    '''

    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 128
    config.depths = (2, 2, 12, 2)
    config.num_heads = (4, 4, 8, 16)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (32, 256, 384)
    return config

def get_3DMPLDSmall_config():
    '''

    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 48
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 4, 4)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (32, 256, 384)
    return config

def get_3DMPLDTiny_config():
    '''

    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 6
    config.depths = (2, 2, 4, 2)
    config.num_heads = (2, 2, 4, 4)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (32, 256, 384)
    return config
