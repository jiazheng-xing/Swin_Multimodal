import torch

def inflate_weights(pretrained, model, window_size, patch_size):
    """Inflate the swin2d parameters to swin3d.

    The differences between swin3d and swin2d mainly lie in an extra
    axis. To utilize the pretrained parameters in 2d model,
    the weight of swin2d models should be inflated to fit in the shapes of
    the 3d counterpart.

    Args:
        logger (logging.Logger): The logger used to print
            debugging infomation.
    """
    checkpoint = torch.load(pretrained, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1, 1,
                                                                                                      patch_size[
                                                                                                          0], 1,
                                                                                                      1) / \
                                            patch_size[0]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        L2 = (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        wd = window_size[0]
        if nH1 != nH2:
            print(f"Error in loading {k}, passing")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                    size=(2 * window_size[1] - 1, 2 * window_size[2] - 1),
                    mode='bicubic')
                relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2,
                                                                                                               L2).permute(
                    1, 0)
        state_dict[k] = relative_position_bias_table_pretrained.repeat(2 * wd - 1, 1)

    return  state_dict