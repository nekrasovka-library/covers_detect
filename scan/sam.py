# from segment_anything import SamPredictor, sam_model_registry
from mobile_sam import sam_model_registry, SamPredictor
import torch
import numpy as np

def get_best_predict(masks, scores):
    max_index = np.argmax(scores)
    return masks[max_index], scores[max_index], scores

def binary_mask_to_bw_image(binary_mask):
    bw_image = binary_mask.astype(np.uint8) * 255
    return bw_image

class Sam:
    """
    Wrapper class for using SAM models.
    """
    def __init__(self, checkpoint_path, model_type="vit_t", transform = None):
        """
        Initialize SAM model.

        Args:
            checkpoint_path (str): Path to the model checkpoint.
            model_type (str): Type of SAM model to use. vit_t is sam tiny https://github.com/ChaoningZhang/MobileSAM
            transform (callable): Optional transform to be applied to input images.
        """
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.model.to(device=self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)

    def model_res(self, img):  # img is a PIL object
        """
        Get encoded embeddings from the SAM model.

        Args:
            img (PIL.Image): Input image.

        Returns:
            torch.Tensor: Encoded embeddings.
        """
        # img_tensor = self.transform(img) if self.transform else img
        enc_embeddings = self.predictor.features
        return enc_embeddings
    
    def process_image(self, input_point, input_label):
        """
        Process an image using SAM model.

        Args:
            input_point: Input point.
            input_label: Input label.

        Returns:
            np.ndarray: Postprocessed image.
            float: Best score.
        """

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        best_mask, best_score, scores = get_best_predict(masks, scores)
        postprocessed_img = binary_mask_to_bw_image(best_mask)
        return postprocessed_img, best_score
