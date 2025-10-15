import os
import numpy as np
import torch
import trimesh
from PIL import Image
from omegaconf import OmegaConf
import cv2

from ..modules.bbox_gen.models.autogressive_bbox_gen import BboxGen
from ..modules.part_synthesis.process_utils import save_parts_outputs
from ..modules.inference_utils import load_img_mask, prepare_bbox_gen_input, prepare_part_synthesis_input, gen_mesh_from_bounds, vis_voxel_coords, merge_parts
from ..modules.part_synthesis.pipelines import OmniPartImageTo3DPipeline
from ..modules.label_2d_mask.label_parts import (
    prepare_image, 
    get_sam_mask, 
    get_mask, 
    clean_segment_edges,
    resize_and_pad_to_square,
    size_th as DEFAULT_SIZE_TH
)
from ..modules.label_2d_mask.visualizer import Visualizer


def save_state_data(state, save_dir,threshold=2000):
    """
    保存 state 中的非字符串数据到磁盘
    
    Args:
        state (dict): 包含状态信息的字典
        save_dir (str): 保存目录路径
    """

    os.makedirs(save_dir, exist_ok=True)
    image_array = np.array(state["image"])
    
    image_pil = Image.fromarray(image_array.astype(np.uint8))
    image_pil.save(os.path.join(save_dir, f"{state['img_name']}_image.png"))

    np.save(os.path.join(save_dir, f"{state['img_name']}_image.npy"), image_array)

    group_ids_array = np.array(state["group_ids"])
    np.save(os.path.join(save_dir, f"{state['img_name']}_group_ids.npy"), group_ids_array)

    original_group_ids_array = np.array(state["original_group_ids"])
    np.save(os.path.join(save_dir, f"{state['img_name']}_original_group_ids.npy"), original_group_ids_array)

    np.save(os.path.join(save_dir, f"{state['img_name']}_threshold.npy"), np.array(threshold))
    print(f"State data saved to {save_dir}")

def load_state_data(state, load_dir):
    """
    从磁盘加载 state 数据
    
    Args:
        state (dict): 原始 state 字典
        load_dir (str): 加载目录路径
        
    Returns:
        dict: 更新后的 state 字典
    """
    # 加载 image 数据
    image_path = os.path.join(load_dir, f"{state['img_name']}_image.npy")
    if os.path.exists(image_path):
        state["image"] = np.load(image_path).tolist()
    
    # 加载 group_ids 数据
    group_ids_path = os.path.join(load_dir, f"{state['img_name']}_group_ids.npy")
    if os.path.exists(group_ids_path):
        state["group_ids"] = np.load(group_ids_path).tolist()
    
    # 加载 original_group_ids 数据
    orig_group_ids_path = os.path.join(load_dir, f"{state['img_name']}_original_group_ids.npy")
    if os.path.exists(orig_group_ids_path):
        state["original_group_ids"] = np.load(orig_group_ids_path).tolist()

    threshold_path = os.path.join(load_dir, f"{state['img_name']}_threshold.npy")
    if os.path.exists(threshold_path):
        state["threshold"] = np.load(threshold_path).item()  # 使用 .item() 获取标量值
    return state



def apply_base_model(part_synthesis_ckpt,partfield_encoder_path,bbox_gen_ckpt,dino_path,config_path,current_path,dino_l_path):
    # load part_synthesis model
    OmniPartImageTo3DPipeline.dino=dino_path
    OmniPartImageTo3DPipeline.dino_moudel=os.path.join(current_path,"facebookresearch/dinov2")
    part_synthesis_pipeline = OmniPartImageTo3DPipeline.from_pretrained(part_synthesis_ckpt,dino_path,os.path.join(current_path,"facebookresearch/dinov2"))

    
    print("[INFO] PartSynthesis model loaded")

    # load bbox_gen model
    bbox_gen_config = OmegaConf.load(config_path).model.args
    bbox_gen_config.partfield_encoder_path = partfield_encoder_path
    bbox_gen_model = BboxGen(bbox_gen_config,os.path.join(current_path,"OmniPart/configs/dinov2-with-registers-large"),ckpt=dino_l_path)
    
    bbox_gen_model.load_state_dict(torch.load(bbox_gen_ckpt,weights_only=False), strict=False)
   
    bbox_gen_model.eval().half()
    print("[INFO] BboxGen model loaded")

    return part_synthesis_pipeline,bbox_gen_model

def process_image(image_path, threshold, sam_mask_generator,rmbg_model,user_dir,DEVICE):
    """Process image and generate initial segmentation"""
    

    #user_dir = os.path.join(TMP_ROOT, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    img_name = os.path.basename(image_path).split(".")[0]
    
    size_th = threshold
    
    img = Image.open(image_path).convert("RGB")
    processed_image = prepare_image(img, rmbg_net=rmbg_model.to(DEVICE))
        
    processed_image = resize_and_pad_to_square(processed_image)
    white_bg = Image.new("RGBA", processed_image.size, (255, 255, 255, 255))
    white_bg_img = Image.alpha_composite(white_bg, processed_image.convert("RGBA"))
    image = np.array(white_bg_img.convert('RGB'))
    
    rgba_path = os.path.join(user_dir, f"{img_name}_processed.png")
    processed_image.save(rgba_path)
    
    print("Generating raw SAM masks without post-processing...")
    raw_masks = sam_mask_generator.generate(image)
    
    raw_sam_vis = np.copy(image)
    raw_sam_vis = np.ones_like(image) * 255
    
    sorted_masks = sorted(raw_masks, key=lambda x: x["area"], reverse=True)
    
    for i, mask_data in enumerate(sorted_masks):
        if mask_data["area"] < size_th:
            continue
            
        color_r = (i * 50 + 80) % 256
        color_g = (i * 120 + 40) % 256
        color_b = (i * 180 + 20) % 256
        color = np.array([color_r, color_g, color_b])
        
        mask = mask_data["segmentation"]
        raw_sam_vis[mask] = color
    
    visual = Visualizer(image)
    
    group_ids, pre_merge_im = get_sam_mask(
        image, 
        sam_mask_generator, 
        visual, 
        merge_groups=None, 
        rgba_image=processed_image,
        img_name=img_name,
        save_dir=user_dir,
        size_threshold=size_th
    )
    
    pre_merge_path = os.path.join(user_dir, f"{img_name}_mask_pre_merge.png")
    
    pre_merge_image=Image.fromarray(pre_merge_im)
    pre_merge_image.save(pre_merge_path)
    pre_split_vis = np.ones_like(image) * 255  
    
    unique_ids = np.unique(group_ids)
    unique_ids = unique_ids[unique_ids >= 0]  
    
    for i, unique_id in enumerate(unique_ids):
        color_r = (i * 50 + 80) % 256
        color_g = (i * 120 + 40) % 256
        color_b = (i * 180 + 20) % 256
        color = np.array([color_r, color_g, color_b])
        
        mask = (group_ids == unique_id)
        pre_split_vis[mask] = color
        
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0:
            center_y = int(np.mean(y_indices))
            center_x = int(np.mean(x_indices))
            cv2.putText(pre_split_vis, str(unique_id), 
                        (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    pre_split_path = os.path.join(user_dir, f"{img_name}_pre_split.png")
    Image.fromarray(pre_split_vis).save(pre_split_path)
    print(f"Pre-split segmentation (before disconnected parts handling) saved to {pre_split_path}")
    
    get_mask(group_ids, image, ids=2, img_name=img_name, save_dir=user_dir)
    
    init_seg_path = os.path.join(user_dir, f"{img_name}_mask_segments_2.png")
    
    seg_img = Image.open(init_seg_path)
    if seg_img.mode == 'RGBA':
        white_bg = Image.new('RGBA', seg_img.size, (255, 255, 255, 255))
        seg_img = Image.alpha_composite(white_bg, seg_img)
        seg_img.save(init_seg_path)
    
    state = {
        "image": image.tolist(),
        "processed_image": rgba_path,
        "group_ids": group_ids.tolist() if isinstance(group_ids, np.ndarray) else group_ids,
        "original_group_ids": group_ids.tolist() if isinstance(group_ids, np.ndarray) else group_ids,
        "img_name": img_name,
        "pre_split_path": pre_split_path, 
    }
    
    return init_seg_path, pre_merge_image, state

def apply_merge(merge_input,sam_mask_generator, state, user_dir,threshold):
    """Apply merge parameters and generate merged segmentation"""
 
    
    if not state:
        return None, None, state

    #user_dir = os.path.join(TMP_ROOT, str(req.session_hash))
        
    # Convert back from list to numpy array
    image = np.array(state["image"])
    # Use original group IDs instead of the most recent ones
    group_ids = np.array(state["original_group_ids"])
    img_name = state["img_name"]
    
    # Load processed image from path
    processed_image = Image.open(state["processed_image"])
    
    # Display the original IDs before merging, SORTED for easier reading
    unique_ids = np.unique(group_ids)
    unique_ids = unique_ids[unique_ids >= 0]  # Exclude background
    print(f"Original segment IDs (used for merging): {sorted(unique_ids.tolist())}")
    
    # Parse merge groups
    merge_groups = None
    try:
        if merge_input:
            merge_groups = []
            group_sets = merge_input.split(';')
            for group_set in group_sets:
                ids = [int(x) for x in group_set.split(',')]
                if ids:
                    # Validate if these IDs exist in the segmentation
                    existing_ids = [id for id in ids if id in unique_ids]
                    missing_ids = [id for id in ids if id not in unique_ids]
                    
                    if missing_ids:
                        print(f"Warning: These IDs don't exist in the segmentation: {missing_ids}")
                    
                    # Only add group if it has valid IDs
                    if existing_ids:
                        merge_groups.append(ids)
                        print(f"Valid merge group: {ids} (missing: {missing_ids if missing_ids else 'none'})")
                    else:
                        print(f"Skipping merge group with no valid IDs: {ids}")
            
            print(f"Using merge groups: {merge_groups}")
    except Exception as e:
        print(f"Error parsing merge groups: {e}")
        return None, None, state
    
    # Initialize visualizer
    visual = Visualizer(image)
    
    # Generate merged segmentation starting from original IDs
    # Add skip_split=True to prevent splitting after merging
    new_group_ids, merged_im = get_sam_mask(
        image, 
        sam_mask_generator, 
        visual, 
        merge_groups=merge_groups, 
        existing_group_ids=group_ids,
        rgba_image=processed_image,
        skip_split=True, 
        img_name=img_name,
        save_dir=user_dir,
        size_threshold=threshold
    )
    
    # Display the new IDs after merging for future reference
    new_unique_ids = np.unique(new_group_ids)
    new_unique_ids = new_unique_ids[new_unique_ids >= 0]  # Exclude background
    print(f"New segment IDs (after merging): {new_unique_ids.tolist()}")
    
    # Clean edges
    new_group_ids = clean_segment_edges(new_group_ids)
    
    # Save merged segmentation visualization
    get_mask(new_group_ids, image, ids=3, img_name=img_name, save_dir=user_dir)
    
    # Path to merged segmentation
    merged_seg_path = os.path.join(user_dir, f"{img_name}_mask_segments_3.png")

    save_mask = new_group_ids + 1
    save_mask = save_mask.reshape(518, 518, 1).repeat(3, axis=-1)
    try:
        save_mask_path=os.path.join(user_dir, f"{img_name}_mask.exr")
        cv2.imwrite(save_mask_path, save_mask.astype(np.float32))
    except:
        try:
            save_mask_path = os.path.join(user_dir, f"{img_name}_mask.npy")
            np.save(save_mask_path, save_mask.astype(np.float32))   
        except:
            save_mask_path=os.path.join(user_dir, f"{img_name}_mask.png")
            save_mask_normalized = (save_mask / np.max(save_mask) * 255).astype(np.uint8)
            cv2.imwrite(save_mask_path, save_mask_normalized) 

    # Update state with the new group IDs but keep original IDs unchanged
    state["group_ids"] = new_group_ids.tolist() if isinstance(new_group_ids, np.ndarray) else new_group_ids
    state["save_mask_path"] = save_mask_path
    print(f"Saved mask to {save_mask_path}")
    return merged_seg_path, state

def explode_mesh(mesh, explosion_scale=0.4):    

    if isinstance(mesh, trimesh.Scene):
        scene = mesh
    elif isinstance(mesh, trimesh.Trimesh):
        print("Warning: Single mesh provided, can't create exploded view")
        scene = trimesh.Scene(mesh)
        return scene
    else:
        print(f"Warning: Unexpected mesh type: {type(mesh)}")
        scene = mesh

    if len(scene.geometry) <= 1:
        print("Only one geometry found - nothing to explode")
        return scene
    
    print(f"[EXPLODE_MESH] Starting mesh explosion with scale {explosion_scale}")
    print(f"[EXPLODE_MESH] Processing {len(scene.geometry)} parts")
    
    exploded_scene = trimesh.Scene()
    
    part_centers = []
    geometry_names = []
    
    for geometry_name, geometry in scene.geometry.items():
        if hasattr(geometry, 'vertices'):
            transform = scene.graph[geometry_name][0]
            vertices_global = trimesh.transformations.transform_points(
                geometry.vertices, transform)
            center = np.mean(vertices_global, axis=0)
            part_centers.append(center)
            geometry_names.append(geometry_name)
            print(f"[EXPLODE_MESH] Part {geometry_name}: center = {center}")
    
    if not part_centers:
        print("No valid geometries with vertices found")
        return scene
    
    part_centers = np.array(part_centers)
    global_center = np.mean(part_centers, axis=0)
    
    print(f"[EXPLODE_MESH] Global center: {global_center}")
    
    for i, (geometry_name, geometry) in enumerate(scene.geometry.items()):
        if hasattr(geometry, 'vertices'):
            if i < len(part_centers):
                part_center = part_centers[i]
                direction = part_center - global_center
                
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-6:
                    direction = direction / direction_norm
                else:
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)
                
                offset = direction * explosion_scale
            else:
                offset = np.zeros(3)
            
            original_transform = scene.graph[geometry_name][0].copy()
            
            new_transform = original_transform.copy()
            new_transform[:3, 3] = new_transform[:3, 3] + offset
            
            exploded_scene.add_geometry(
                geometry, 
                transform=new_transform, 
                geom_name=geometry_name
            )
            
            print(f"[EXPLODE_MESH] Part {geometry_name}: moved by {np.linalg.norm(offset):.4f}")
    
    print("[EXPLODE_MESH] Mesh explosion complete")
    return exploded_scene


def prepare_inputs(state,part_synthesis_pipeline,bbox_gen_model,user_dir,seed,output_dir,device):
  
    img_path = state["processed_image"]
    mask_path = state["save_mask_path"]
    #user_dir = os.path.join(TMP_ROOT, str(req.session_hash))
    img_white_bg, img_black_bg, ordered_mask_input, img_mask_vis = load_img_mask(img_path, mask_path)
    img_mask_vis.save(os.path.join(user_dir, "img_mask_vis.png"))
    
    if not os.path.isfile(os.path.join(user_dir, "voxel_coords.npy")) or  not os.path.isfile(os.path.join(user_dir, "voxel_coords_vis.ply")):
        part_synthesis_pipeline.to(device) # to save memory
        voxel_coords = part_synthesis_pipeline.get_coords(img_black_bg, num_samples=1, seed=seed, sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5})
        
        voxel_coords = voxel_coords.cpu().numpy()
        np.save(os.path.join(user_dir, "voxel_coords.npy"), voxel_coords)
        voxel_coords_ply = vis_voxel_coords(voxel_coords)
        voxel_coords_ply.export(os.path.join(user_dir, "voxel_coords_vis.ply"))
        print("[INFO] Voxel coordinates saved")
    else:
        print("[INFO] Voxel coordinates already exists")
    bbox_gen_input = prepare_bbox_gen_input(os.path.join(user_dir, "voxel_coords.npy"), img_white_bg, ordered_mask_input)

    if not os.path.exists(os.path.join(user_dir, "bboxes.npy")) or not os.path.exists(os.path.join(user_dir, "bboxes_vis.glb")):
        bbox_gen_model.to(device)
        bbox_gen_output = bbox_gen_model.generate(bbox_gen_input)

        np.save(os.path.join(user_dir, "bboxes.npy"), bbox_gen_output['bboxes'][0])
        bboxes_vis = gen_mesh_from_bounds(bbox_gen_output['bboxes'][0])
        bboxes_vis.export(os.path.join(user_dir, "bboxes_vis.glb"))
        print("[INFO] BboxGen output saved")
    else:
        print("[INFO] BboxGen output already exists")
    part_synthesis_input = prepare_part_synthesis_input(os.path.join(user_dir, "voxel_coords.npy"), os.path.join(user_dir, "bboxes.npy"), ordered_mask_input)
    part_synthesis_input.update({"img_black_bg":img_black_bg,"output_dir":output_dir})
    bbox_gen_model.to("cpu")
    return part_synthesis_input



def inference_omnipart(part_synthesis_pipeline,part_synthesis_input,num_inference_steps,guidance_scale,simplify_ratio,device):
    part_synthesis_pipeline.to(device)
    output_dir=part_synthesis_input["output_dir"]
    part_synthesis_output = part_synthesis_pipeline.get_slat(
        part_synthesis_input["img_black_bg"], 
        part_synthesis_input['coords'], 
        [part_synthesis_input['part_layouts']], 
        part_synthesis_input['masks'],
        seed=part_synthesis_input["seed"],
        slat_sampler_params={"steps": num_inference_steps, "cfg_strength": guidance_scale},
        formats=['mesh', 'gaussian', 'radiance_field'],
        preprocess_image=False,
    )

    save_parts_outputs(
        part_synthesis_output, 
        output_dir=output_dir, 
        simplify_ratio=simplify_ratio, 
        save_video=False,
        save_glb=True,
        textured=False,
        prefix=part_synthesis_input.get("img_name","omni"),
    )
    mesh_texture_path,mesh_segment_path=merge_parts(output_dir,part_synthesis_input.get("img_name","omni"))
    print("[INFO] PartSynthesis output saved")

    return mesh_texture_path,mesh_segment_path

# if __name__ == "__main__":
#     device = "cuda"

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image-input", type=str, required=True)
#     parser.add_argument("--mask-input", type=str, required=True)
#     parser.add_argument("--output-root", type=str, default="./output")
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--num-inference-steps", type=int, default=25)
#     parser.add_argument("--guidance-scale", type=float, default=7.5)
#     parser.add_argument("--simplify_ratio", type=float, default=0.3)
#     parser.add_argument("--partfield_encoder_path", type=str, default="ckpt/model_objaverse.ckpt")
#     parser.add_argument("--bbox_gen_ckpt", type=str, default="ckpt/bbox_gen.ckpt")
#     parser.add_argument("--part_synthesis_ckpt", type=str, default="ckpt/part_synthesis")
#     args = parser.parse_args()

#     os.makedirs(args.output_root, exist_ok=True)
#     output_dir = os.path.join(args.output_root, args.image_input.split("/")[-1].split(".")[0])
#     os.makedirs(output_dir, exist_ok=True)

#     torch.manual_seed(args.seed)

    # # load part_synthesis model
    # part_synthesis_pipeline = OmniPartImageTo3DPipeline.from_pretrained(args.part_synthesis_ckpt)
    # part_synthesis_pipeline.to(device)
    # print("[INFO] PartSynthesis model loaded")

    # # load bbox_gen model
    # bbox_gen_config = OmegaConf.load("configs/bbox_gen.yaml").model.args
    # bbox_gen_config.partfield_encoder_path = args.partfield_encoder_path
    # bbox_gen_model = BboxGen(bbox_gen_config)
    # bbox_gen_model.load_state_dict(torch.load(args.bbox_gen_ckpt), strict=False)
    # bbox_gen_model.to(device)
    # bbox_gen_model.eval().half()
    # print("[INFO] BboxGen model loaded")
    
    # img_white_bg, img_black_bg, ordered_mask_input, img_mask_vis = load_img_mask(args.image_input, args.mask_input)
    # img_mask_vis.save(os.path.join(output_dir, "img_mask_vis.png"))

    # voxel_coords = part_synthesis_pipeline.get_coords(img_black_bg, num_samples=1, seed=args.seed, sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5})
    # voxel_coords = voxel_coords.cpu().numpy()
    # np.save(os.path.join(output_dir, "voxel_coords.npy"), voxel_coords)
    # voxel_coords_ply = vis_voxel_coords(voxel_coords)
    # voxel_coords_ply.export(os.path.join(output_dir, "voxel_coords_vis.ply"))
    # print("[INFO] Voxel coordinates saved")

    # bbox_gen_input = prepare_bbox_gen_input(os.path.join(output_dir, "voxel_coords.npy"), img_white_bg, ordered_mask_input)
    # bbox_gen_output = bbox_gen_model.generate(bbox_gen_input)
    # np.save(os.path.join(output_dir, "bboxes.npy"), bbox_gen_output['bboxes'][0])
    # bboxes_vis = gen_mesh_from_bounds(bbox_gen_output['bboxes'][0])
    # bboxes_vis.export(os.path.join(output_dir, "bboxes_vis.glb"))
    # print("[INFO] BboxGen output saved")

    #part_synthesis_input = prepare_part_synthesis_input(os.path.join(output_dir, "voxel_coords.npy"), os.path.join(output_dir, "bboxes.npy"), ordered_mask_input)
    # part_synthesis_output = part_synthesis_pipeline.get_slat(
    #     img_black_bg, 
    #     part_synthesis_input['coords'], 
    #     [part_synthesis_input['part_layouts']], 
    #     part_synthesis_input['masks'],
    #     seed=args.seed,
    #     slat_sampler_params={"steps": args.num_inference_steps, "cfg_strength": args.guidance_scale},
    #     formats=['mesh', 'gaussian', 'radiance_field'],
    #     preprocess_image=False,
    # )
    # save_parts_outputs(
    #     part_synthesis_output, 
    #     output_dir=output_dir, 
    #     simplify_ratio=args.simplify_ratio, 
    #     save_video=False,
    #     save_glb=True,
    #     textured=False,
    # )
    # merge_parts(output_dir)
    # print("[INFO] PartSynthesis output saved")
