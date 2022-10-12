"""
    split dataset indices to train.txt and test.txt
"""
import os
import random

def main():
    # find the voc dataset indices
    voc_anno_path = "../FASTER RCNN/VOCdevkit/VOC2012/Annotations/"
    files_name = [file_name.split(".")[0] for file_name in os.listdir(voc_anno_path)]
    files_num = len(files_name)

    #split val and train indices
    random.seed(27149)
    val_rate = 0.5
    val_index = random.sample(range(files_num), k = int(files_num * val_rate))
    train_files, val_files = [], []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)
    
    #writer splited indices to text file
    try:
        train_f = open("./dataset/train.txt", "x")
        val_f = open("./dataset/val.txt", "x")
        train_f.write("\n".join(train_files))
        val_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
    


if __name__ == "__main__":
    main()