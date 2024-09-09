from PIL import Image
from rembg import new_session, remove

class BackgroundRemover:
    def __init__(self, model_name="isnet-general-use"):#u2net
        self.model_name = model_name
        self.session = new_session(self.model_name)

    def remove_background(self, input_image):
        if not isinstance(input_image, Image.Image):
            raise ValueError("Input image must be a PIL.Image object.")
        
        try:
            output_image = remove(input_image, session=self.session)
            return output_image
        except Exception as e:
            print(f"Error removing background: {str(e)}")
            return None
