 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from .model_loader_utils import  tensor2image,phi2narry,nomarl_upscale
from .OmniPart.scripts.inference_omnipart import (
    inference_omnipart,prepare_inputs,apply_base_model,save_state_data,load_state_data,
    process_image,apply_merge)
import folder_paths
import comfy
import node_helpers
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from pathlib import PureWindowsPath
from datetime import datetime
import nodes
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_OmniPart_current_path = os.path.join(folder_paths.models_dir, "OmniPart")
if not os.path.exists(weigths_OmniPart_current_path):
    os.makedirs(weigths_OmniPart_current_path)

weigths_dinov2_current_path = os.path.join(folder_paths.models_dir, "dinov2")
if not os.path.exists(weigths_dinov2_current_path):
    os.makedirs(weigths_dinov2_current_path)

folder_paths.add_model_folder_path("OmniPart", weigths_OmniPart_current_path) #  OmniPart dir
folder_paths.add_model_folder_path("dinov2", weigths_dinov2_current_path)

class OmniPart_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="OmniPart_SM_Model",
            display_name="OmniPart_SM_Model",
            category="OmniPart",
            inputs=[
                io.String.Input("custom_path", multiline=False,default="omnipart/OmniPart",),
                io.Combo.Input("box_ckpt",options= ["none"] + [i for i in folder_paths.get_filename_list("OmniPart") if "box" in i.lower()]),
                io.Combo.Input("partfield_encoder",options= ["none"] + [i for i in folder_paths.get_filename_list("OmniPart") if "encoder" in i.lower()]),
                io.Combo.Input("dino",options= ["none"] + folder_paths.get_filename_list("dinov2") ),
                io.Combo.Input("dino_l",options= ["none"] + folder_paths.get_filename_list("dinov2") ),
            ],
            outputs=[
                io.Custom("OmniPart_SM_Model").Output(),
                ],
            )
    @classmethod
    def execute(cls, custom_path,box_ckpt,partfield_encoder,dino,dino_l) -> io.NodeOutput:
        part_synthesis_ckpt=PureWindowsPath(custom_path).as_posix() if custom_path else None
        bbox_gen_ckpt=folder_paths.get_full_path("OmniPart", box_ckpt) if box_ckpt != "none" else None
        partfield_encoder_path=folder_paths.get_full_path("OmniPart", partfield_encoder) if box_ckpt != "none" else None
        dino_path=folder_paths.get_full_path("dinov2", dino) if dino != "none" else None
        dino_l_path=folder_paths.get_full_path("dinov2", dino_l) if dino_l != "none" else None
        assert bbox_gen_ckpt is not None and partfield_encoder is not None and dino_path is not None and dino_l_path is not None, "Please select the BboxGen  and partfield_encoder and dino and  dino l checkpoint file"
        config_path=os.path.join(node_cr_path,"OmniPart/configs/bbox_gen.yaml")
        if part_synthesis_ckpt is None:
            part_synthesis_ckpt="omnipart/OmniPart"
        part_synthesis_pipeline,bbox_gen_model=apply_base_model(part_synthesis_ckpt,partfield_encoder_path,bbox_gen_ckpt,dino_path,config_path,node_cr_path,dino_l_path)
        model={"part_synthesis_pipeline":part_synthesis_pipeline,"bbox_gen_model":bbox_gen_model}
        return io.NodeOutput(model)
    

class OmniPart_SM_PreModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OmniPart_SM_PreModel",
            display_name="OmniPart_SM_PreModel",
            category="OmniPart",
            inputs=[
                io.Combo.Input("rmgb",options= ["none"] + folder_paths.get_filename_list("OmniPart")),
                io.Combo.Input("sam",options= ["none"] + [i for i in folder_paths.get_filename_list("OmniPart") if "sam" in i.lower()]),
                ],
            outputs=[
                io.Custom("OmniPart_SM_SamModel").Output(display_name="sam_model"),
                io.Custom("OmniPart_SM_SegModel").Output(display_name="rmbg_model"),
                     ],
        )
    @classmethod
    def execute(cls, rmgb,sam,) -> io.NodeOutput:
        sam_ckpt_path=folder_paths.get_full_path("OmniPart", sam) if sam != "none" else None
        rmgb_ckpt_path=folder_paths.get_full_path("OmniPart", rmgb) if rmgb != "none" else None
        assert sam_ckpt_path is not None and rmgb_ckpt_path is not None, "Please select the SAM and RMBG 2.0 checkpoint file"

        print("Loading SAM model...")
        from  segment_anything import SamAutomaticMaskGenerator, build_sam #seg
        sam_model = build_sam(checkpoint=sam_ckpt_path)
     
        print("Loading BriaRMBG 2.0 model...")
        from transformers import AutoConfig
        from .OmniPart.configs.RMBG.birefnet import BiRefNet
        rmgb_repo=os.path.join(node_cr_path,"OmniPart/configs/RMBG")
        config = AutoConfig.from_pretrained(rmgb_repo, trust_remote_code=True)
        rmbg_model = BiRefNet(config=config)
        sd=comfy.utils.load_torch_file(rmgb_ckpt_path)
        rmbg_model.load_state_dict(sd,strict=False)
        del sd
        rmbg_model.eval()
       # rmbg_model = AutoModelForImageSegmentation.from_pretrained(rmgb_repo, trust_remote_code=True)
        
        return io.NodeOutput (sam_model,rmbg_model)

class OmniPart_SM_PreImg(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OmniPart_SM_PreImg",
            display_name="OmniPart_SM_PreImg",
            category="OmniPart",
            inputs=[
                io.Custom("OmniPart_SM_SamModel").Input("sam_model"),
                io.Custom("OmniPart_SM_SegModel").Input("rmbg_model"),
                io.Image.Input("image"),
                io.Int.Input("threshold", default=1600, min=600, max=4000,step=200),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output( display_name="dir_prefix"),
                ],
        )
    @classmethod
    def execute(cls, sam_model,rmbg_model,image,threshold) -> io.NodeOutput:
        phi_in=tensor2image(image)
        
        dir_prefix = datetime.now().strftime("%y%m%d%H%M%S")[-6:]      
        user_dir=os.path.join(folder_paths.get_output_directory(), f"OmniPart/temp_{dir_prefix}")
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
       
        image_path=os.path.join(user_dir,f"img_{dir_prefix}.png")
        phi_in.save(image_path)

        sam_model.to(device)
        from  segment_anything import SamAutomaticMaskGenerator
        sam_mask_generator = SamAutomaticMaskGenerator(sam_model)
        rmbg_model.to(device)

        init_seg_path, pre_merge_image, state=process_image(image_path, threshold, sam_mask_generator,rmbg_model,user_dir,device)
        save_state_data(state, user_dir,threshold)
        
        sam_model.to("cpu")
        del sam_mask_generator
        rmbg_model.to("cpu")

        torch.cuda.empty_cache()
        image=phi2narry(pre_merge_image)
        return io.NodeOutput(image, dir_prefix)


class OmniPart_SM_MergeImg(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OmniPart_SM_MergeImg",
            display_name="OmniPart_SM_MergeImg",
            category="OmniPart",
            inputs=[
                io.Custom("OmniPart_SM_SamModel").Input("sam_model"), 
                io.String.Input( "dir_prefix", multiline=False,default="",),
                io.String.Input( "merge_input", multiline=False,default="1,3;2,4",),
                io.Image.Input("image",optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="state"),
                ],
        )
    @classmethod
    def execute(cls, sam_model,dir_prefix,merge_input,image=None) -> io.NodeOutput:
        if image is not None:
            image=tensor2image(image)
        if not dir_prefix:
            dir_prefix_list=extract_loadimage_filenames(recent_workflow_data_sm)
            if not dir_prefix_list:
                raise ValueError("dir_prefix is required if no active workflow found input a valid dir_prefix")
            else:
                print(dir_prefix_list)
                for prefix in dir_prefix_list:
                    if isinstance(prefix, str) and "_" in prefix:
                        dir_prefix_ = prefix.split("_",1)[0]
                        print(dir_prefix_)
                        if len(dir_prefix_)==6:
                            dir_prefix = dir_prefix_
                            break
                        else:
                            continue
                            
        else:
            if len(dir_prefix)!=6:
                raise ValueError("dir_prefix is required a number of 6 digits ")
        user_dir=os.path.join(folder_paths.get_output_directory(), f"OmniPart/temp_{dir_prefix}")
        img_name=f"img_{dir_prefix}"
        state={
            "processed_image": os.path.join(user_dir, f"{img_name}_processed.png"),
            "img_name": img_name,
            "pre_split_path": os.path.join(user_dir, f"{img_name}_pre_split.png"), 
            "user_dir": user_dir,
        }
        state=load_state_data(state, user_dir)

        sam_model.to(device)
        from  segment_anything import SamAutomaticMaskGenerator
        sam_mask_generator = SamAutomaticMaskGenerator(sam_model)
        merged_seg_path, cond=apply_merge(merge_input, sam_mask_generator,state, user_dir,int(state.get("threshold",2000)))
        cond.update(state)
        sam_model.to("cpu")
        del sam_mask_generator
        torch.cuda.empty_cache()
        return io.NodeOutput(state)


class OmniPart_SM_Prebox(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OmniPart_SM_Prebox",
            display_name="OmniPart_SM_Prebox",
            category="OmniPart",
            inputs=[
                io.Custom("OmniPart_SM_Model").Input("model"),
                io.Conditioning.Input("state"),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
            ],
            outputs=[
                io.Conditioning.Output(display_name="cond"),
                ],
        )
    @classmethod
    def execute(cls, model, state,seed) -> io.NodeOutput:
        cond=prepare_inputs(state,model["part_synthesis_pipeline"],model["bbox_gen_model"],user_dir=state.get("user_dir"),seed=seed,output_dir=folder_paths.get_output_directory(),device=device) 
        cond.update({"seed":seed})
        return io.NodeOutput(cond,)
       

class OmniPart_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OmniPart_SM_KSampler",
            display_name="OmniPart_SM_KSampler",
            category="OmniPart",
            inputs=[
                io.Custom("OmniPart_SM_Model").Input("model"),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01,),
                io.Conditioning.Input("cond"),
                io.Float.Input("simplify_ratio", default=0.3, min=0.0, max=1.0, step=0.1,display_mode=io.NumberDisplay.slider),
                
            ],
            outputs=[
                io.String.Output(display_name="scene_texture"),
                io.String.Output(display_name="mesh_segment"),
            ],
        )
    
    @classmethod
    def execute(cls, model, steps, cfg, cond,simplify_ratio ) -> io.NodeOutput:
        scene_texture,mesh_segment=inference_omnipart(model["part_synthesis_pipeline"],cond,steps,cfg,simplify_ratio,device)
        return io.NodeOutput(scene_texture,mesh_segment)

from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/OmniPart_SM_Extension")
async def get_hello(request):
    return web.json_response("OmniPart_SM_Extension")

class OmniPart_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            OmniPart_SM_Model,
            OmniPart_SM_PreModel,
            OmniPart_SM_PreImg,
            OmniPart_SM_MergeImg,
            OmniPart_SM_Prebox,
            OmniPart_SM_KSampler,
        ]
async def comfy_entrypoint() -> OmniPart_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return OmniPart_SM_Extension()


recent_workflow_data_sm = None

def on_prompt_handler(json_data):
    global recent_workflow_data_sm
    recent_workflow_data_sm = json_data
    return json_data

PromptServer.instance.add_on_prompt_handler(on_prompt_handler)

def extract_loadimage_filenames(json_data):
    filenames = [] 
    if not json_data or not isinstance(json_data, dict):
        return filenames
        
    nodes_data = json_data.get("prompt", {}).get("nodes", [])
    if not nodes_data and "nodes" in json_data:
        nodes_data = json_data.get("nodes", [])
    

    for node in nodes_data:
        if node.get("type") == "LoadImage":
            widgets_values = node.get("widgets_values", [])
            if widgets_values and len(widgets_values) > 0:
                filename = widgets_values[0]
                if isinstance(filename, str):
                    filenames.append(filename)
    
    return filenames


