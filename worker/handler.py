import datetime
import io
import logging
import os

import torch
from PIL import Image
from torchvision import transforms as T
from ts.torch_handler.vision_handler import VisionHandler

from model import MobileNet

logger = logging.getLogger(__name__)


class CustomHandler(VisionHandler):

    image_processing = T.Compose([
        T.ToTensor(),
    ])

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.py file")
        self.device = torch.device('cpu')
        net = MobileNet(in_channels=1, out_cls=10, pretrained=0)
        chk = torch.load(model_pt_path, map_location='cpu')
        net.load_state_dict(chk['state_dict'])
        return net

    def preprocess(self, data):
        print(f"Datetime preprocess: {datetime.datetime.now()}")
        images = []
        for row in data:
            image = row.get("data") or row.get("body") or row.get("images")
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                print(f"Image shape: {image.size}")
                image = self.image_processing(image)
            else:
                image = torch.FloatTensor(image)
            images.append(image)

        images = torch.stack(images).to(self.device)
        return images


