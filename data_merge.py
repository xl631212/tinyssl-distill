import os
import shutil

real_root = "/home/xuyingl/iccv_workshop/data/real"
synth_root = "/home/xuyingl/iccv_workshop/data/synth"
all_root = "/home/xuyingl/iccv_workshop/data/all"

os.makedirs(all_root, exist_ok=True)

for cname in os.listdir(real_root):
    src_real = os.path.join(real_root, cname)
    src_synth = os.path.join(synth_root, cname)
    dst = os.path.join(all_root, cname)
    
    # 只合并 synth 里有的类别
    if os.path.exists(src_synth):
        os.makedirs(dst, exist_ok=True)
        # 合并真实图片
        for f in os.listdir(src_real):
            shutil.copy(os.path.join(src_real, f), os.path.join(dst, f))
        # 合并合成图片
        for f in os.listdir(src_synth):
            shutil.copy(os.path.join(src_synth, f), os.path.join(dst, f))
        print(f"已合并类别: {cname}")
    else:
        print(f"synth 中不存在类别: {cname}，已跳过")

print("合并完成，仅合并了 synth 里也有的类别。")
