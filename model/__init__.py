from model.resnet import resnet
from config import parse_arguments
model_dict = {
    'resnet': resnet
}

def create_model(model_name, num_classes, pretrained=False):
    args = parse_arguments()
    return model_dict[model_name](num_classes=num_classes, pretrained=pretrained)
