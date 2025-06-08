import argparse
import math
import os
import gc
import torch
import torchvision
from torch import optim
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import dlib
from argparse import Namespace

from style_criteria.clip_loss import CLIPLoss
from style_criteria.id_loss import IDLoss
from mapper.training.train_utils import STYLESPACE_DIMENSIONS
from style_models.stylegan2.model import Generator
import clip
from style_util import ensure_checkpoint_exists

# Add these imports for e4e encoding
from encoder4editing.models.psp import pSp
from encoder4editing.utils.alignment import align_face

STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def encode_real_image(image_path, e4e_model_path, shape_predictor_path=None):
    """
    Encode a real image using e4e encoder to get W+ latent codes
    """
    # Load e4e model
    ckpt = torch.load(e4e_model_path, map_location="cpu")
    opts = ckpt["opts"]
    opts["checkpoint_path"] = e4e_model_path
    opts = Namespace(**opts)
    
    e4e_net = pSp(opts)
    e4e_net.eval()
    e4e_net = e4e_net.cuda()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load and preprocess image
    if shape_predictor_path and os.path.exists(shape_predictor_path):
        # Align face if shape predictor is available
        shape_predictor = dlib.shape_predictor(shape_predictor_path)
        aligned_image = align_face(filepath=image_path, predictor=shape_predictor)
        input_image = aligned_image.resize((256, 256))
    else:
        # Just resize if no alignment
        input_image = Image.open(image_path).convert('RGB')
        input_image = input_image.resize((256, 256))
    
    # Transform and encode
    transformed_image = transform(input_image)
    
    with torch.no_grad():
        images, latents = e4e_net(
            transformed_image.unsqueeze(0).cuda().float(), 
            randomize_noise=False, 
            return_latents=True
        )

    # get result
    result_latents = latents[0].detach().clone()
    
    # clean up
    del images, latents, transformed_image
    del e4e_net, ckpt
    if shape_predictor is not None:
        del shape_predictor
    
    # clean up
    gc.collect()
    torch.cuda.empty_cache()
    
    return result_latents, input_image
    
    

def main(args):
    ensure_checkpoint_exists(args.ckpt)
    if args.use_break_down_expression:
        text_inputs_list = [torch.cat([clip.tokenize(component_description)]).cuda() for component_description in args.component_descriptions]
    else:
        text_inputs = torch.cat([clip.tokenize(args.description)]).cuda()
    os.makedirs(args.results_dir, exist_ok=True)

    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)

    # Get latent code from real image using e4e encoding
    if args.image_path:
        print(f"Encoding real image: {args.image_path}")
        latent_code_init, processed_image = encode_real_image(
            args.image_path, 
            args.e4e_model_path,
            getattr(args, 'shape_predictor_path', None)
        )
        print("Successfully encoded real image to W+ latent space")
    elif args.latent_path:
        latent_code_init = torch.load(args.latent_path).cuda()
    elif args.mode == "edit":
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                        truncation=args.truncation, truncation_latent=mean_latent)
    else:
        latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)
    
    with torch.no_grad():
        img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

    if args.work_in_stylespace:
        with torch.no_grad():
            _, _, latent_code_init = g_ema([latent_code_init], input_is_latent=True, return_latents=True)
        latent = [s.detach().clone() for s in latent_code_init]
        for c, s in enumerate(latent):
            if c in STYLESPACE_INDICES_WITHOUT_TORGB:
                s.requires_grad = True
    else:
        latent = latent_code_init.detach().clone()
        latent.requires_grad = True

    clip_loss = CLIPLoss(args)
    id_loss = IDLoss(args)

    print("Initiated loss functions")

    if args.work_in_stylespace:
        optimizer = optim.Adam(latent, lr=args.lr)
    else:
        optimizer = optim.Adam([latent], lr=args.lr)

    pbar = tqdm(range(args.step))
    print("Start optimization")

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)

        if args.use_break_down_expression:
            c_loss = sum([clip_loss(img_gen, text_inputs_list[c]) for c in range(len(text_inputs_list))]) / len(text_inputs_list)
        else:
            c_loss = clip_loss(img_gen, text_inputs)

        if args.id_lambda > 0:
            i_loss = id_loss(img_gen, img_orig)[0]
        else:
            i_loss = 0

        if args.mode == "edit":
            if args.work_in_stylespace:
                l2_loss = sum([((latent_code_init[c] - latent[c]) ** 2).sum() for c in range(len(latent_code_init))])
            else:
                l2_loss = ((latent_code_init - latent) ** 2).sum()
            loss = c_loss + args.l2_lambda * l2_loss + args.id_lambda * i_loss
        else:
            loss = c_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
        if args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
            with torch.no_grad():
                img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)

            torchvision.utils.save_image(img_gen, f"results/{str(i).zfill(5)}.jpg", normalize=True, value_range=(-1, 1))

    if args.mode == "edit":
        final_result = torch.cat([img_orig, img_gen])
    else:
        final_result = img_gen

    return final_result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default="a person with purple hair", help="the text that guides the editing/generation")
    parser.add_argument("--ckpt", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--step", type=int, default=300, help="number of optimization steps")
    parser.add_argument("--mode", type=str, default="edit", choices=["edit", "free_generation"], help="choose between edit an image an generate a free one")
    parser.add_argument("--l2_lambda", type=float, default=0.008, help="weight of the latent distance (used for editing only)")
    parser.add_argument("--id_lambda", type=float, default=0.000, help="weight of id loss (used for editing only)")
    parser.add_argument("--latent_path", type=str, default=None, help="starts the optimization from the given latent code if provided. Otherwose, starts from"
                                                                      "the mean latent in a free generation, and from a random one in editing. "
                                                                      "Expects a .pt format")
    parser.add_argument("--truncation", type=float, default=0.7, help="used only for the initial latent vector, and only when a latent code path is"
                                                                      "not provided")
    parser.add_argument('--work_in_stylespace', default=False, action='store_true')
    parser.add_argument("--save_intermediate_image_every", type=int, default=20, help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--ir_se50_weights', default='../pretrained_models/model_ir_se50.pth', type=str,
                             help="Path to facial recognition network used in ID loss")
    
    # Add arguments for real image encoding
    parser.add_argument("--image_path", type=str, default=None, help="path to input image to encode")
    parser.add_argument("--e4e_model_path", type=str, default="../pretrained_models/e4e_ffhq_encode.pt", help="path to e4e encoder model")
    parser.add_argument("--shape_predictor_path", type=str, default="../pretrained_models/shape_predictor_68_face_landmarks.dat", help="path to dlib shape predictor for face alignment")
    parser.add_argument('--use_break_down_expression', default=False, action='store_true', help="use component descriptions instead of single description")
    parser.add_argument("--component_descriptions", type=str, nargs='+', default=[], help="list of component descriptions when using break down expression")

    args = parser.parse_args()

    result_image = main(args)

    torchvision.utils.save_image(result_image.detach().cpu(), os.path.join(args.results_dir, "final_result.jpg"), normalize=True, scale_each=True, range=(-1, 1))


