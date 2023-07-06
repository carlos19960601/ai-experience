from io import BytesIO

import torch
import werkzeug
from flask_restful import Resource, reqparse
from PIL import Image
from transformers import (AutoTokenizer, VisionEncoderDecoderModel,
                          ViTImageProcessor)


class Image2Text(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('image', required=True, type=werkzeug.datastructures.FileStorage, location='files')
        args = parser.parse_args()
        uploadImage = args["image"]

        return {"description": self.predict_step(uploadImage)}
    
    def predict_step(self, uploadImage):
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    
        images = []
        i_image = Image.open(uploadImage)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds[0]
