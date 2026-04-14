import open_clip
from .point_encoder import PointcloudEncoder
import timm


def build_text_encoder():    
    clip_model_type = "EVA02-E-14-plus"
    pretrained = "./pretrained_weight/clip_used_in_Uni3D/open_clip_pytorch_model.bin"
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name=clip_model_type, pretrained=pretrained)

    return clip_model



def build_pc_encoder(args):

    pretrained_pc = ''
    drop_path_rate = 0.0

    pc_encoder_type = getattr(args, 'pc_encoder_type', 'small')

    if pc_encoder_type == "giant":
        pc_model = "eva_giant_patch14_560"
        args.pc_feat_dim = 1408
    elif pc_encoder_type == "large":
        pc_model = "eva02_large_patch14_448"
        args.pc_feat_dim = 1024
    elif pc_encoder_type == "base":
        pc_model = "eva02_base_patch14_448"
        args.pc_feat_dim = 768
    elif pc_encoder_type == "small":
        pc_model = "eva02_small_patch14_224"
        args.pc_feat_dim = 384
    elif pc_encoder_type == "tiny":
        pc_model = "eva02_tiny_patch14_224"
        args.pc_feat_dim = 192


    # create transformer blocks for point cloud via timm
    point_transformer = timm.create_model(pc_model, checkpoint_path=pretrained_pc, drop_path_rate= drop_path_rate)

    # create whole point cloud encoder
    point_encoder = PointcloudEncoder(point_transformer, args)

    return point_encoder