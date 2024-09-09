import PIL.Image

# from carvekit.api.interface import Interface
# from carvekit.ml.wrap.fba_matting import FBAMatting
# from carvekit.ml.wrap.u2net import U2NET
# from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
# from carvekit.pipelines.postprocessing import MattingMethod
# from carvekit.pipelines.preprocessing import PreprocessingStub
# from carvekit.trimap.generator import TrimapGenerator

# import onnxruntime
from typing import Dict

import numpy as np

def softmax(x, axis=0, eps=1e-10):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=axis) + eps) # only difference

class ImageProcessor:
    """
    scales the image by the biggest side of the image into 512 size
    and other side proportionally
    """
    def __init__(self, max_side=512):
        self.max_side = max_side
        
    def process(self, image: PIL.Image) -> PIL.Image:
        width, height = image.size
        rate_w = width / self.max_side
        rate_h = height / self.max_side
        rate = max(rate_w, rate_h)
        
        if rate > 1:
            newsize = (int(width/rate), int(height/rate))
            image_new = image.resize(newsize)
        else:
            newsize = (width, height)
            image_new = image
        return image_new
    
    def __call__(self, img: PIL.Image) -> PIL.Image:
        return self.process(img)


# class ModelBG:
#     def __init__(self, batch_size=1, device='cpu', input_tensor_size=630):
#         self.batch_size = batch_size
#         # seg_net = U2NET(device=device, batch_size=batch_size)
#         seg_net = TracerUniversalB7(device=device, batch_size=batch_size)
#         fba = FBAMatting(device=device, input_tensor_size=input_tensor_size, batch_size=batch_size)
#         trimap = TrimapGenerator(prob_threshold = 100)
#         preprocessing = PreprocessingStub()
#         postprocessing = MattingMethod(matting_module=fba, trimap_generator=trimap, device=device)
#         self.interface = Interface(pre_pipe=preprocessing, post_pipe=postprocessing, seg_pipe=seg_net)
    
#     def forward(self, img: PIL.Image) -> PIL.Image:
#         image_bg = self.interface([img])[0]
#         return image_bg
        
    
#     def __call__(self, img: PIL.Image) -> PIL.Image:
#         return self.forward(img)
    

# class OnnxRunner:
#     def __init__(self, onnx_model_path,
#                  providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']):
#         self.onnx_model_path = onnx_model_path
#         self.ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
        
#     def __call__(self, x: np.ndarray) -> Dict:
#         ort_inputs = {self.ort_session.get_inputs()[0].name: x}
#         ort_outs = self.ort_session.run(None, ort_inputs)
#         outputs = softmax(ort_outs[0], axis=1)
#         return outputs
    


