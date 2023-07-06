from flask_restful import Resource, reqparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


class Translation(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("texts", type=list, required=True, location="json")
        args = parser.parse_args()

        model_checkpoint = "Helsinki-NLP/opus-mt-en-zh"
        translator = pipeline("translation", model=model_checkpoint)
        translatedText = translator(args["texts"])
        return translatedText
       


class Translation2(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("text", required=True, location="json")
        args = parser.parse_args()

        model_checkpoint = "Helsinki-NLP/opus-mt-en-zh"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        translator = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)
        x=["If nothing is detected and there is a config.json file, it’s assumed the library is transformers.","By looking into the presence of files such as *.nemo or *saved_model.pb*, the Hub can determine if a model is from NeMo or Keras."]
        translatedText = translator(args["text"], max_length=450)
        return translatedText