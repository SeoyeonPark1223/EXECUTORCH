import torch
from torchvision.models import resnet18, ResNet18_Weights
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

def export_resnet18():
    """
    1. Export ResNet18 model
    2. Optimize for target hardware (backend = XNNPACK)
    3. Save for deployment
    """

    model_path = "./model/model_outputs/resnet18.pte"

    # 1. Export ResNet18 model
    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    example_inputs = (torch.randn(1, 3, 224, 224),)
    exported_program = torch.export.export(model, example_inputs)

    # 2. Optimize for target hardware (backend = XNNPACK)
    program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    # 3. Save for deployment
    with open(model_path, "wb") as f:
        f.write(program.buffer)
    
    return model_path

if __name__ == "__main__":
    export_resnet18()