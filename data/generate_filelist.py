import os
import sys
import glob


if __name__ == "__main__":

    root = "data/HHM50K"
    writer = open("data/HHM50K.txt", "w")

    images = sorted(glob.glob(os.path.join(root, "images/*.jpg"))) 
    alphas = sorted(glob.glob(os.path.join(root, "alphas/*.png"))) 
    fgs = sorted(glob.glob(os.path.join(root, "foregrounds/*.jpg"))) 
    bgs = sorted(glob.glob(os.path.join(root, "backgrounds/*.jpg"))) 

    assert len(images) == len(alphas)
    assert len(images) == len(fgs)
    assert len(images) == len(bgs)

    for img, pha, fg, bg in zip(images, alphas, fgs, bgs):
        img_name = img.split('/')[-1][:-4]
        pha_name = pha.split('/')[-1][:-4]
        fg_name = fg.split('/')[-1][:-4]
        bg_name = bg.split('/')[-1][:-4]
        assert img_name == pha_name
        assert img_name == fg_name
        assert img_name == bg_name
        writer.write(f"{img},{pha},{fg},{bg}\n")


    root = "data/HHM2K"
    writer = open("data/HHM2K.txt", "w")

    images = sorted(glob.glob(os.path.join(root, "images/*.jpg"))) 
    alphas = sorted(glob.glob(os.path.join(root, "alphas/*.png"))) 

    assert len(images) == len(alphas)

    for img, pha in zip(images, alphas):
        img_name = img.split('/')[-1][:-4]
        pha_name = pha.split('/')[-1][:-4]
        assert img_name == pha_name
        writer.write(f"{img},{pha}\n")
