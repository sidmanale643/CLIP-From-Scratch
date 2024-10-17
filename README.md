# DistilCLIP: Vision Transformer + DistilBERT CLIP Implementation

DistilCLIP is a from-scratch implementation of a CLIP-like model, using a Vision Transformer (ViT) as the image encoder and a pretrained DistilBERT as the text encoder. This model was trained on the Naruto BLIP Captions dataset for 25 epochs to understand multimodal representations of anime images and their corresponding captions.

DistilCLIP is inspired by OpenAI's CLIP but substitutes the original encoders with a Vision Transformer (ViT) for processing images and DistilBERT for text processing. This model allows for efficient and scalable multimodal learning.

## Architecture

- **Image Encoder**: Vision Transformer (ViT)
- **Text Encoder**: Pretrained DistilBERT
- **Multimodal Projection**: Linear layers that project both image and text features into a shared latent space, following a CLIP-like architecture.

![image](https://github.com/user-attachments/assets/922d11d0-2a9f-4afb-98c5-2443420f57b6)

## Dataset

"jxie/flickr8k" on HuggingFace

The Flickr 8K Dataset consists of general image and caption pairs.
- **Number of Images**: 1,224
- **Caption Format**: Descriptive text describing the scene, characters, or context of the image.
- **Data Augmentation**: Minimal augmentations were applied to images, including resizing and normalization.

## Training Details

The model was trained for 25 epochs on the Naruto BLIP Captions Dataset.

Key details:
- **Optimizer**: AdamW
- **Learning Rate**: 5e-5 (with cosine annealing schedule)
- **Batch Size**: 64
- **Loss Function**: Contrastive loss, aiming to align similar image-text pairs in the latent space
- **Training Hardware**: Tesla T4 GPU
- **Epochs**: 25 (Underfitted , more epochs needed)

## Results

![image](https://github.com/user-attachments/assets/fcfe091b-5572-4bc3-b52c-7b82d3b13507)





## Installation

To set up the DistilCLIP project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/sidmanale643/Distil-CLIP.git
   cd Distil-CLIP
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```


## Contributing

Contributions to DistilCLIP are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- OpenAI for the original CLIP model
- Hugging Face for the DistilBERT implementation
- Lambdalabs

## Contact

For any questions or feedback, please open an issue in this repository or contact sidmanale643@gmail.com.
