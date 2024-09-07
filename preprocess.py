# PREPROCESSING
# voglio che tutte le immagini si avvicinino alla dimensione di 64x64
    # eseguo una resize proporzionale che mi permetta di avvicinarmi
    # successivamente farò un crop centrale

import os
import cv2
import math
from torchvision import transforms

final_size = (64, 64)
# Path (non sono divise tra train e val, uso random_split in seguito)
folders_path = 'dataset/garbage_classification'

def preprocess(output_folder_name):
    folders = os.listdir(folders_path)

    for type in folders:
        folder_path = os.path.join(folders_path, type)
        files = os.listdir(folder_path)

        for file in files:
            # Carica l'immagine e ottieni le dimensioni
            img = cv2.imread(os.path.join(folder_path, file))

            if img is not None:

                scale = max(final_size) / min(img.shape[:2])
        
                # Ridimensiona mantenendo le proporzioni
                img = cv2.resize(img, (int(math.ceil(img.shape[1] * scale)), int(math.ceil(img.shape[0] * scale))))
                
                cropped =   img[img.shape[0]//2 - final_size[0]//2 : img.shape[0]//2 + final_size[0]//2,
                                img.shape[1]//2 - final_size[1]//2 : img.shape[1]//2 + final_size[1]//2]

                output_dir = f"{output_folder_name}/{type}"
                output_path = output_dir + f"/{file.split('.jpg')[0]}_preprocessed.jpg"

                if os.path.exists(output_dir):
                    cv2.imwrite(output_path, cropped)
                else:
                    os.makedirs(output_dir)
                    cv2.imwrite(output_path, cropped)
        print(f"finished with {type}")
    print("Done with Preprocessing!")

def preprocess_single(folder_path, image_name):

    img = cv2.imread(os.path.join(folder_path, image_name))

    if img is not None:

        scale = max(final_size) / min(img.shape[:2])

        # Ridimensiona mantenendo le proporzioni
        img = cv2.resize(img, (int(math.ceil(img.shape[1] * scale)), int(math.ceil(img.shape[0] * scale))))
        
        cropped =   img[img.shape[0]//2 - final_size[0]//2 : img.shape[0]//2 + final_size[0]//2,
                        img.shape[1]//2 - final_size[1]//2 : img.shape[1]//2 + final_size[1]//2]

        output_dir = f"{folder_path}/{image_name.split(f'.')[0]}_preprocessed." + "png"
        cv2.imwrite(output_dir, cropped)
        return output_dir


