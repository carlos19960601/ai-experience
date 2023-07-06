from flask import Flask
from flask_restful import Api

from image2text import Image2Text
from text2image import Text2Image
from translation import Translation

app = Flask(__name__)
api = Api(app)


api.add_resource(Translation, '/translation')
api.add_resource(Image2Text, '/image2text')
api.add_resource(Text2Image, '/text2image')

if __name__ == '__main__':
    app.run(debug=True)