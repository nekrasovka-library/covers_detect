import cv2
import numpy as np
import matplotlib.pyplot as plt

class EdgeDetection:
    def __init__(self,
                 proto_path,
                 model_path):
        
        self.proto_path = proto_path
        self.model_path = model_path
        self.net = self.load_model()


    class CropLayer(object):
        """
         There is a Crop layer that the HED network uses which is not implemented by 
         default so we need to provide our own implementation of this layer.
         Without the crop layer, the final result will be shifted to the right and bottom
         cropping part of the image
        """
        def __init__(self, params, blobs):
            # initialize our starting and ending (x, y)-coordinates of the crop
            self.startX = 0
            self.startY = 0
            self.endX = 0
            self.endY = 0

        def getMemoryShapes(self, inputs):
            # the crop layer will receive two inputs -- we need to crop
            # the first input blob to match the shape of the second one,
            # keeping the batch size and number of channels
            (inputShape, targetShape) = (inputs[0], inputs[1])
            (batchSize, numChannels) = (inputShape[0], inputShape[1])
            (H, W) = (targetShape[2], targetShape[3])

            # compute the starting and ending crop coordinates
            self.startX = int((inputShape[3] - targetShape[3]) / 2)
            self.startY = int((inputShape[2] - targetShape[2]) / 2)
            self.endX = self.startX + W
            self.endY = self.startY + H

            # return the shape of the volume (we'll perform the actual
            # crop during the forward pass
            return [[batchSize, numChannels, H, W]]

        def forward(self, inputs):
            # use the derived (x, y)-coordinates to perform the crop
            return [inputs[0][:, :, self.startY:self.endY,
                    self.startX:self.endX]]

    def load_model(self):
        import os
        print(os.getcwd())
        print(self.proto_path, self.model_path)
        net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
        cv2.dnn_registerLayer("Crop", self.CropLayer)
        return net

    def preprocess_image(self, image):
        (H, W) = image.shape[:2]
        mean_pixel_values = np.average(image, axis=(0, 1))
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                                     mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                                     # mean=(105, 117, 123),
                                     swapRB=True, crop=False)
        return blob

    def detect_edges(self, image):
        blob = self.preprocess_image(image)
        self.net.setInput(blob)
        hed = self.net.forward()
        hed = hed[0, 0, :, :]
        hed = (255 * hed).astype("uint8")
        return hed


