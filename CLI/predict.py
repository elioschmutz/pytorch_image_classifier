from model_trainer import ModelTrainer
from PIL import Image
import argparse
import json
import math
import numpy as np
import torch


def process_image(image):
    # Resize
    width, height = image.size
    ratio = min(width, height) / 256
    image.thumbnail((width / ratio, height / ratio))

    # Crop
    width, height = image.size
    width_distance = (width - 255) / 2
    height_distance = (height - 255) / 2

    left = math.floor(width_distance)
    right = math.floor(width - width_distance)
    upper = math.floor(height_distance)
    lower = math.floor(height - height_distance)
    image = image.crop((left, upper, right, lower))

    # Convert to an ndarray
    np_image = np.array(image)

    # Squosh color values from 0-255 to 0-1
    np_image = np_image / 255

    # Normalize
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))

    tensor_image = torch.tensor(np_image)

    # Image tensor needs to be in type torch.FloatTensor
    tensor_image = tensor_image.float()

    return tensor_image


def predict(image_path, model, topk=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = Image.open(image_path)
    model.eval()
    model.to(device)
    with torch.no_grad():
        image = process_image(image)

        # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
        image = image.unsqueeze_(0)

        image = image.to(device)
        probs = torch.exp(model(image))
        top_probs, top_classes = probs.topk(topk, dim=1)
        return top_probs[0].tolist(), [str(i) for i in top_classes[0].tolist()]


def main():
    parser = argparse.ArgumentParser(description='Image model trainer')
    parser.add_argument('image_path', type=str)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('category_names', type=str)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'])
    parser.add_argument('--top_k', type=int, default=1)


    args = parser.parse_args()

    model = ModelTrainer(device=args.device).load_model(args.checkpoint_path, True).model

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict(args.image_path, model, args.top_k)
    idx_to_class = {str(j): int(i) for i, j in model.class_to_idx.items()}
    cat_ids = [str(idx_to_class[i]) for i in classes]
    class_names = [cat_to_name[cat_id].capitalize() for cat_id in cat_ids]

    for (classname, prob) in zip(class_names, probs):
        print(f"{classname}: {(prob * 100):.1f}%")

if __name__ == "__main__":
    main()
