import os

import numpy as np
import torch


def export_deployment_bundle_to_onnx(deployment_bundle, onnx_path, opset_version):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model = deployment_bundle['model'].cpu().eval()
    example_input = deployment_bundle['support_data'][:1].cpu()
    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=opset_version,
    )
    if deployment_bundle['deployment_type'] == 'encoder_with_prototypes':
        prototype_path = os.path.splitext(onnx_path)[0] + '_prototypes.npz'
        np.savez(
            prototype_path,
            prototypes=deployment_bundle['prototypes'].cpu().numpy(),
            selected_labels=np.asarray(deployment_bundle['selected_labels']),
        )
        deployment_bundle['prototype_path'] = prototype_path
