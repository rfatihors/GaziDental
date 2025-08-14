# import torch
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
#
# checkpoint = "./checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"
# predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
#
# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#     predictor.set_image(<your_image>)
#     masks, _, _ = predictor.predict(<input_prompts>)

from ultralytics import SAM

# Load a model
model = SAM("sam2_l.pt")

# Display model information (optional)
model.info()

# Run inference
results3 = model("local/gummy_smile_project/gum/gum/images/IMG_2632.jpg")

results = model("local/gummy_smile_project/gum/gum/images/IMG_111.jpg", bboxes=[100, 100, 200, 200])

# Segment with point prompt
results2 = model("local/gummy_smile_project/gum/gum/images/IMG_111.jpg", points=[150, 150], labels=[1])

results3[0].show()
results2[0].show()
results[0].show()