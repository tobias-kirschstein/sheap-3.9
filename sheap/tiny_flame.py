from pathlib import Path

import torch
import torch.nn.functional as F
from roma import rotvec_to_rotmat
from torch import nn
from typing import Union


class TinyFlame(nn.Module):
    v_template: torch.Tensor
    J_regressor: torch.Tensor
    shapedirs: torch.Tensor
    posedirs: torch.Tensor
    weights: torch.Tensor
    faces: torch.Tensor
    kintree: torch.Tensor

    def __init__(
        self,
        ckpt: Union[Path, str],
        eyelids_ckpt: Union[Path, str, None] = None,
    ) -> None:
        """A tiny version of the FLAME model that is compatible with ONNX."""
        super().__init__()

        # Load the FLAME model weights
        ckpt = Path(ckpt).expanduser()
        data = torch.load(ckpt)

        for name, tensor in data.items():
            self.register_buffer(name, tensor)

        # Load the eyelids blendshapes if provided
        if eyelids_ckpt is not None:
            eyelids_ckpt = Path(eyelids_ckpt).expanduser()
            eyelids_data = torch.load(eyelids_ckpt)

            self.register_buffer("eyelids_dirs", eyelids_data)
        else:
            self.eyelids_dirs = None

        # To work around the limitation of TorchDynamo, we need to convert kinematic tree to a list,
        # such that it is treated as a constant.
        self.parents = self.kintree[0].tolist()

    def forward(
        self,
        shape: torch.Tensor,
        expression: torch.Tensor,
        pose: torch.Tensor,
        translation: torch.Tensor,
        eyelids: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """Convert FLAME parameters to coordinates of FLAME vertices.

        Args:
        - shape (torch.Tensor): Shape parameters of the FLAME model with shape (N, 300).
        - expression (torch.Tensor): Expression parameters of the FLAME model with shape (N, 100).
        - pose (torch.Tensor): Pose parameters of the FLAME model as 3x3 matrices with shape (N, 5, 3, 3).
            It is the concatenation of torso pose (global rotation), neck pose, jaw pose,
            and left/right eye poses.
        - translation (torch.Tensor): Global translation parameters of the FLAME model with shape (N, 3).
        - eyelids (torch.Tensor): Eyelids blendshape parameters with shape (N, 2).

        Returns:
        - vertices (torch.Tensor): The vertices of the FLAME model with shape (N, V, 3).
        """
        # Some common variables
        batch_size = shape.shape[0]
        num_joints = len(self.parents)

        # Step1: compute T per equations (2)-(5) in the paper
        # Compute the shape offsets from the shape and the expression parameters
        shape_expr = torch.cat([shape, expression], -1)
        shape_expr_offsets = (self.shapedirs @ shape_expr.t()).permute(2, 0, 1)

        # Get the vertex offsets due to pose blendshapes
        pose_features = pose[:, 1:, :, :] - torch.eye(3, device=pose.device)
        pose_features = pose_features.view(batch_size, -1)
        pose_offsets = (self.posedirs @ pose_features.t()).permute(2, 0, 1)

        # Add offsets to the template mesh to get T
        shaped_vertices = self.v_template.expand_as(shape_expr_offsets) + shape_expr_offsets
        if eyelids is not None and self.eyelids_dirs is not None:
            shaped_vertices = shaped_vertices + (self.eyelids_dirs @ eyelids.t()).permute(2, 0, 1)
        shaped_vertices_with_pose_correction = shaped_vertices + pose_offsets

        # Step2: compute the joint locations per equation (1) in the paper
        # Get the joint locations with the joint regressor
        joint_locations = self.J_regressor @ shaped_vertices

        # Step3: compute the final mesh vertices per equation (1) in the paper using standard LBS functions.
        # Find the transformation for: unposed FLAME -> joints' local coordinate systems -> posed FLAME
        relative_joint_locations = (
            joint_locations[:, 1:, :] - joint_locations[:, self.parents[1:], :]
        )
        relative_joint_locations = torch.cat(
            [joint_locations[:, :1, :], relative_joint_locations], dim=1
        )
        relative_joint_locations_homogeneous = F.pad(relative_joint_locations, (0, 1), value=1)

        # joint -> parent joint transformations
        joint_to_parent_transformations = torch.cat(
            [
                F.pad(pose, (0, 0, 0, 1), value=0),
                relative_joint_locations_homogeneous.unsqueeze(-1),
            ],
            dim=-1,
        )

        joint_to_posed_transformations_ = [joint_to_parent_transformations[:, 0, :, :]]

        # joint -> posed FLAME transformations
        for i in range(1, num_joints):
            parent_joint = self.parents[i]

            current_joint_to_posed_transformation = (
                joint_to_posed_transformations_[parent_joint]
                @ joint_to_parent_transformations[:, i, :, :]
            )

            joint_to_posed_transformations_.append(current_joint_to_posed_transformation)

        joint_to_posed_transformations = torch.stack(joint_to_posed_transformations_, dim=1)

        # Unposed FLAME -> joints' local coordinate systems -> posed FLAME transformations
        unposed_to_posed_transformations = joint_to_posed_transformations - F.pad(
            joint_to_posed_transformations @ F.pad(joint_locations, (0, 1), value=0).unsqueeze(-1),
            (3, 0),
            value=0,
        )

        # Scale rotations and translations by the blend weights
        final_transformations = (self.weights @ unposed_to_posed_transformations.flatten(2)).view(
            batch_size, -1, 4, 4
        )

        # Apply the transformations to the posed vertices T
        shaped_vertices_with_pose_correction_homogeneous = F.pad(
            shaped_vertices_with_pose_correction, (0, 1), value=1
        )
        posed_vertices = (
            final_transformations @ shaped_vertices_with_pose_correction_homogeneous.unsqueeze(-1)
        )[..., :3, 0] + translation.unsqueeze(1)

        return posed_vertices


def pose_components_to_rotmats(predictions):
    """
    predictions should contain these 5 keys:
    'torso_pose', 'neck_pose', 'jaw_pose', 'eye_l_pose', 'eye_r_pose'
    Each of these is expected to be of shape (N, 3) representing rotation vectors.
    This function converts them to rotation matrices and stacks them into a tensor of shape (N, 5, 3, 3).
    """
    pose = torch.stack(
        [
            predictions["torso_pose"],
            predictions["neck_pose"],
            predictions["jaw_pose"],
            predictions["eye_l_pose"],
            predictions["eye_r_pose"],
        ],
        dim=1,
    )
    pose = pose.view(-1, 3)
    pose = rotvec_to_rotmat(pose)
    return pose.view(-1, 5, 3, 3)
