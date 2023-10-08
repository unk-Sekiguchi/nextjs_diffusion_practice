import torch
import datetime
from diffusers import StableDiffusionPipeline
# from optimum.intel.openvino import OVStableDiffusionPipeline


# trinartモデル
model_id = "naclbit/trinart_stable_diffusion_v2" 

#GPUチェック
if  torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# モデルの読み込み
pipe = StableDiffusionPipeline.from_pretrained(
                                    model_id,
                                    use_auth_token=True,
                                    revision="diffusers-60k")

# デバイスの設定
pipe = pipe.to(device)

# 画像生成
async def generate_image(payload):
    generator_list =[]

    # シードを設定
    for i in range(payload.count):
        generator_list.append(torch.Generator(device).manual_seed(payload.seedList[i]))
    
    # 画像生成
    images_list = pipe(
        [payload.prompt] * payload.count,
        width = payload.width,
        height = payload.height,
        negative_prompt = [payload.negative] * payload.count,
        guidance_scale = payload.scale,
        num_inference_steps=payload.steps,
        generator=generator_list,
    )
    print(images_list)
    images = []
    for i, image in enumerate(images_list["images"]):
        file_name = (
            "./outputs/image_"
            + datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
            + ".png"
        )
        image.save(file_name)
        images.append(image)
    
    return images