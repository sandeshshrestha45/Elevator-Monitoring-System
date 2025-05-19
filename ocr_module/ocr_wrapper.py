import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from ocr_module.utils import CTCLabelConverter, AttnLabelConverter
from ocr_module.model import Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

opt = {
    "Transformation": "TPS",
    "FeatureExtraction": "ResNet",
    "SequenceModeling": "BiLSTM",
    "Prediction": "Attn",
    "saved_model": "/home/elevatorpi4/project/EMS-phase2-deploy/ocr_module/saved_models/best_accuracy.pth",
    "character": '0123456789abcdefghijklmnopqrstuvwxyz',
    "batch_max_length": 25,
    "imgH": 32,
    "imgW": 100, 
    "rgb": False,
    "sensitive": False,
    "PAD": False,
    "num_fiducial": 20,
    "input_channel": 1,
    "output_channel": 512,
    "hidden_size": 256
}

def initialize_model():
    """Load the OCR model and converter."""
    if 'CTC' in opt["Prediction"]:
        converter = CTCLabelConverter(opt["character"])
    else:
        converter = AttnLabelConverter(opt["character"])

    opt["num_class"] = len(converter.character)
    if opt["rgb"]:
        opt["input_channel"] = 3

    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    model.load_state_dict(torch.load(opt["saved_model"], map_location=device))
    model.eval()

    return model, converter

ocr_model, text_converter = initialize_model()

def preprocess_image(image_path):
    img_ = cv2.imread(image_path, cv2.IMREAD_COLOR if opt["rgb"] else cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite('before_test.jpg', img_)
    img = cv2.transpose(img_)  
    img = cv2.flip(img, flipCode=1) 
    # cv2.imwrite('after_test.jpg', img)
    if img is None:
        raise ValueError(f"Cannot read image from {image_path}")
    if opt["rgb"]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = cv2.resize(img, (opt["imgW"], opt["imgH"]))
    img = img.astype(np.float32) / 255.0
    if opt["rgb"]:
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) → (C, H, W)
    else:
        img = np.expand_dims(img, axis=0)  # (H, W) → (1, H, W)
    img_tensor = torch.FloatTensor(img).unsqueeze(0)  # (1, C, H, W)

    return img_tensor.to(device)

def predict_text(image_path):
    img_tensor = preprocess_image(image_path)
    batch_size = img_tensor.size(0)
    length_for_pred = torch.IntTensor([opt["batch_max_length"]] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, opt["batch_max_length"] + 1).fill_(0).to(device)
    with torch.no_grad():
        if 'CTC' in opt["Prediction"]:
            preds = ocr_model(img_tensor, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            pred_text = text_converter.decode(preds_index, preds_size)
        else:
            preds = ocr_model(img_tensor, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            pred_text = text_converter.decode(preds_index, length_for_pred)

    if 'Attn' in opt["Prediction"]:
        pred_text = [text[:text.find('[s]')] if '[s]' in text else text for text in pred_text]

    return pred_text[0]  


# if __name__ == '__main__':
#     image_file = "C:/Users/Anjil/Documents/shinwa-active/EMS-phase-2/number_segmentation/plate_segmentation_rotated/vid1 (3407).jpg"  
#     result = predict_text(image_file)
#     print(f"Predicted Text: {result}")
