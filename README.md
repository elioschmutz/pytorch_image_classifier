
# Image classifier
Image classifier based on pytorch.

## Prerequisites
- https://pytorch.org/ (use with conda)

### Folder structure
Expected folder structure of the images:

- image-folder/
  - test/
    - cat1/
    - cat2/
    - cat3/
  - valid/
     - cat1/
      - cat2/
      - cat3/

### Category .json
```json
{"cat1": "Foo Bar", "cat2": "John Doe"}
```

## Train
```cli
python train.py ./image-folder
```

## Predict
```cli
python train.py ./path-to-image/image.jpg ./path-to-trainer-checkpoint/model.pth ./path-to-category-json/cat.json
```
