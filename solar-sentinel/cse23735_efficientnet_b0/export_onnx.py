import os
import torch
import numpy as np
from model import create_model
from dataset import NUM_CLASSES, CLASS_NAMES

def export_to_onnx(checkpoint_path='best_efficientnet_b0.pth', onnx_path='efficientnet_b0_solar.onnx'):
    print(f"Exporting {checkpoint_path} to ONNX format ({onnx_path})...")
    
    device = torch.device('cpu')
    model = create_model(num_classes=NUM_CLASSES, pretrained=False)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained model checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        print("Warning: Checkpoint not found. Exporting initial model architecture.")
        
    model.eval()

    # Dummy input matching model input shape: Batch=1, Channels=3, Height=224, Width=224
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    
    input_names = ['input_image']
    output_names = ['output_logits']
    
    dynamic_axes = {
        'input_image': {0: 'batch_size'},
        'output_logits': {0: 'batch_size'}
    }
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    print(f"Successfully exported ONNX model to {os.path.abspath(onnx_path)}")
    print(f"ONNX Model File Size: {os.path.getsize(onnx_path) / (1024 * 1024):.2f} MB")
    print("Class labels order:", CLASS_NAMES)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoint to ONNX format")
    parser.add_argument('--checkpoint', type=str, default='best_efficientnet_b0.pth', help='Path to input PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='efficientnet_b0_solar.onnx', help='Path to output ONNX file')
    args = parser.parse_args()
    
    export_to_onnx(checkpoint_path=args.checkpoint, onnx_path=args.output)
