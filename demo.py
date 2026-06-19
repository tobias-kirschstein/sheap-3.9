import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sheap import inference_images_list, load_sheap_model, render_mesh
from sheap.tiny_flame import TinyFlame, pose_components_to_rotmats

# os.environ["PYOPENGL_PLATFORM"] = "egl"


def create_rendering_image(
    original_image: Image.Image,
    verts: torch.Tensor,
    faces: torch.Tensor,
    c2w: torch.Tensor,
    output_size: int = 512,
) -> Image.Image:
    """
    Create a combined image with original, mesh, and blended views.

    Args:
        original_image: PIL Image of the original frame
        verts: Vertices tensor for a single frame, shape (num_verts, 3)
        faces: Faces tensor, shape (num_faces, 3)
        c2w: Camera-to-world transformation matrix, shape (4, 4)
        output_size: Size of each sub-image in the combined output

    Returns:
        PIL Image with three views side-by-side (original, mesh, blended)
    """
    # Render the mesh
    color, depth = render_mesh(verts=verts, faces=faces, c2w=c2w)

    # Resize original to match output size
    original_resized = original_image.convert("RGB").resize((output_size, output_size))

    # Create blended image (mesh overlaid on original)
    mask = (depth > 0).astype(np.float32)[..., None]
    blended = (np.array(color) * mask + np.array(original_resized) * (1 - mask)).astype(np.uint8)

    # Combine all three images horizontally
    combined = Image.new("RGB", (output_size * 3, output_size))
    combined.paste(original_resized, (0, 0))
    combined.paste(Image.fromarray(color), (output_size, 0))
    combined.paste(Image.fromarray(blended), (output_size * 2, 0))

    return combined


if __name__ == "__main__":
    # Load SHeaP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sheap_model = load_sheap_model(model_type="expressive").to(device)

    # Inference on example images
    folder_containing_images = Path("example_images/")
    image_paths = list(sorted(folder_containing_images.glob("*.jpg")))
    with torch.no_grad():
        predictions = inference_images_list(
            model=sheap_model,
            device=device,
            image_paths=image_paths,
            batch_size=1,
            num_workers=0,
        )

    # Load and infer FLAME with our predicted parameters
    flame_dir = Path("FLAME2020/")
    flame = TinyFlame(flame_dir / "generic_model.pt", eyelids_ckpt=flame_dir / "eyelids.pt")
    verts = flame(
        shape=predictions["shape_from_facenet"],
        expression=predictions["expr"],
        pose=pose_components_to_rotmats(predictions),
        eyelids=predictions["eyelids"],
        translation=predictions["cam_trans"],
    )

    # Render the FLAME mesh for each input image
    c2w = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.float32
    )
    for i_frame in range(verts.shape[0]):
        outpath = image_paths[i_frame].with_name(f"{image_paths[i_frame].name}_rendered.png")
        if outpath.exists():
            outpath.unlink()

        # Load original image
        original = Image.open(image_paths[i_frame])

        # Create combined rendering
        combined = create_rendering_image(
            original_image=original,
            verts=verts[i_frame],
            faces=flame.faces,
            c2w=c2w,
            output_size=512,
        )
        combined.save(outpath)
