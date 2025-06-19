import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

# 路径参数
input_dir = './data/real'
output_dir = './data/synth'
os.makedirs(output_dir, exist_ok=True)

device = "cuda"  # 有N卡的话建议用GPU
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to(device)

for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    synth_class_path = os.path.join(output_dir, class_name)
    os.makedirs(synth_class_path, exist_ok=True)
    for fname in os.listdir(class_path):
        img_path = os.path.join(class_path, fname)
        image = Image.open(img_path).convert("RGB").resize((512, 512))
        for i in range(10):
            # 你可以自定义prompt，也可以直接用类别id或加点风格修饰
            prompt = f"a photo of a {class_name}"
            synth_img = pipe(
                prompt=prompt,
                image=image,
                strength=0.65,  # 控制与原图相似度，越大越多变
                guidance_scale=7.5,  # 控制prompt影响力
                num_inference_steps=50,
            ).images[0]
            synth_img.save(os.path.join(synth_class_path, f"{fname.replace('.jpg','')}_var{i}.jpg"))
        print(f"[{class_name}] {fname} 扩增完毕")
print("所有合成变体已生成！")
