import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from flask_restful import Resource, reqparse


class Text2Image(Resource):
    def post(self):
      parser = reqparse.RequestParser()
      args = parser.parse_args()

      model_id = "stabilityai/stable-diffusion-2-1"
      pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
      pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config) 
      
      prompt = "a photo of an astronaut riding a horse on mars"
      image = pipe(prompt).images[0]  
          
      image.save("astronaut_rides_horse.png")
      return {}       

