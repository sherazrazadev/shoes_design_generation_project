import json
import os
from contextlib import contextmanager
from datetime import datetime
import torch
from modelfiles.Unet import Unet
from modelfiles.Victorai import Victorai

def _create_directory(dir_path):
    original_dir = os.getcwd()
    img_path = os.path.join(original_dir, dir_path, "generated_images")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    elif not len(os.listdir(img_path)) == 0:
        raise FileExistsError(f"The directory {os.path.join(original_dir, img_path)} already exists and is nonempty")

    @contextmanager
    def cm(subdir=""):
        os.chdir(os.path.join(original_dir, dir_path, subdir))
        yield
        os.chdir(original_dir)
    return cm

def _get_best_state_dict(unet_number, files):
    filt_list = list(filter(lambda x: x.startswith(f"unet_{unet_number}"), files))
    min_val = min([i.split("_")[-1].split(".pth")[0] for i in filt_list])
    return list(filter(lambda x: x.endswith(f"{min_val}.pth"), filt_list))[0]

def _read_params(directory, filename):
    with open(os.path.join(directory, "parameters", filename), 'r') as _file:
        return json.loads(_file.read())

def load_params(directory):
    files = os.listdir(os.path.join(directory, "parameters"))
    unets_params_files = sorted(list(filter(lambda x: x.startswith("unet_", ), files)),
                                key=lambda x: int(x.split("_")[1]))
    unets_params = [_read_params(directory, f) for f in unets_params_files]
    imagen_params_files = _read_params(directory, list(filter(lambda x: x.startswith("imagen_"), files))[0])
    return unets_params, imagen_params_files

def _instatiate_victorai(directory):
    unets_params, imagen_params_files = load_params(directory)
    return Imagen(unets=[Unet(**params) for params in unets_params], **imagen_params_files)

def load_victorai(directory):
    map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    victorai = _instatiate_victorai(directory)
    files = os.listdir(os.path.join(directory, "state_dicts"))
    if files != []:
        subdir = "state_dicts"
        num_unets = int(max(set([i.split("_")[1] for i in list(filter(lambda x: x.startswith("unet_"), files))])) + 1
        unet_state_dicts = [list(filter(lambda x: x.startswith(f"unet_{i}"), files))[0] for i in range(num_unets)]
        for idx, file in enumerate(unet_state_dicts):
            pth = os.path.join(directory, f'{subdir}', file)
            victorai.unets[idx].load_state_dict(torch.load(pth, map_location=map_location))
    else:
        subdir = "tmp"
        print(f"\n\"state_dicts\" folder in {directory} is empty, using the most recent checkpoint from \"tmp\".\n")
        files = os.listdir(os.path.join(directory, f"{subdir}"))
        if files == []:
            raise ValueError(f"Both \"/state_dicts\" and \"/tmp\" in {directory} are empty. Train the model to acquire state dictionaries for inference. ")
        num_unets = int(max(set([i.split("_")[1] for i in list(filter(lambda x: x.startswith("unet_"), files))])) + 1
        unet_state_dicts = [list(filter(lambda x: x.startswith(f"unet_{i}"), files))[0] for i in range(num_unets)]
        for idx, file in enumerate(unet_state_dicts):
            pth = os.path.join(directory, f'{subdir}', file)
            victorai.unets[idx].load_state_dict(torch.load(pth, map_location=map_location))
    return victorai

def sample_and_save(captions: list, victorai: Imagen = None, training_directory: str = None, sample_args: dict = {}, save_directory: str = None, filetype: str = "png"):
    assert not (victorai is None and training_directory is None), "Must supply either a training directory or Victorai instance."
    assert (victorai != None) ^ (training_directory != None), "Cannot supply both a Victorai instance and a training directory"
    if save_directory is None:
        save_directory = datetime.now().strftime("generated_images_%Y%m%d_%H%M%S")
    cm = _create_directory(save_directory)
    with cm():
        with open('captions.txt', 'w') as f:
            for caption in captions:
                f.write(f"{caption}\n")
        if training_directory is not None:
            with open('imagen_training_directory.txt', 'w') as f:
                f.write(training_directory)
    if training_directory is not None:
        victorai = load_victorai(training_directory)
    images = victorai.sample(texts=captions, return_pil_images=True, **sample_args)
    with cm("generated_images"):
        for idx, img in enumerate(images):
            img.save(f'image_{idx}.{filetype}')
