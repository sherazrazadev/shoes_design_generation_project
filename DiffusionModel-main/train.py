import os
from datetime import datetime

import torch.utils.data
from torch import optim

from modelfiles.Victorai import Victorai
from modelfiles.Unet import Unet, Base, Super, BaseTest, SuperTest
from modelfiles.generate import load_Victorai, load_params
from modelfiles.t5 import get_encoded_dim
from modelfiles.training import get_victorai_parser, ConceptualCaptions, get_victorai_dl_opts, \
    create_directory, get_model_params, get_model_size, save_training_info, get_default_args, VictoraiTrain, \
    load_restart_training_parameters, load_testing_parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = get_victorai_parser()
parser.add_argument("-ts", "--TIMESTAMP", dest="timestamp", help="Timestamp for training directory", type=str,
                             default=None)
args = parser.parse_args()
timestamp = args.timestamp

if timestamp is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


dir_path = f"./training_{timestamp}"
training_dir = create_directory(dir_path)

if args.RESTART_DIRECTORY is not None:
    args = load_restart_training_parameters(args)
elif args.PARAMETERS is not None:
    args = load_restart_training_parameters(args, justparams=True)

if args.TESTING:
    args = load_testing_parameters(args)
    train_dataset, valid_dataset = ConceptualCaptions(args, smalldata=True)
else:
    train_dataset, valid_dataset = ConceptualCaptions(args, smalldata=False)

dl_opts = {**get_victorai_dl_opts(device), 'batch_size': args.BATCH_SIZE, 'num_workers': args.NUM_WORKERS}
train_dataloader = torch.utils.data.DataLoader(train_dataset, **dl_opts)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **dl_opts)

if args.RESTART_DIRECTORY is None:
    imagen_params = dict(
        image_sizes=(int(args.IMG_SIDE_LEN / 2), args.IMG_SIDE_LEN),
        timesteps=args.TIMESTEPS,
        cond_drop_prob=0.15,
        text_encoder_name=args.T5_NAME
    )

    if args.TESTING:
        # If testing, use tiny victorai for low computational load
        unets_params = [get_default_args(BaseTest), get_default_args(SuperTest)]

    elif not args.PARAMETERS:
        unets_params = [get_default_args(Base), get_default_args(Super)]

    else:

        unets_params, imagen_params = get_model_params(args.PARAMETERS)

    unets = [Unet(**unet_params).to(device) for unet_params in unets_params]

    imagen = Imagen(unets=unets, **imagen_params).to(device)
else:

    orig_train_dir = os.path.join(os.getcwd(), args.RESTART_DIRECTORY)
    unets_params, imagen_params = load_params(orig_train_dir)
    imagen = load_victorai(orig_train_dir).to(device)
    unets = imagen.unets

unets_params = [{**get_default_args(Unet), **i} for i in unets_params]
imagen_params = {**get_default_args(Imagen), **imagen_params}

model_size_MB = get_model_size(imagen)

save_training_info(args, timestamp, unets_params, imagen_params, model_size_MB, training_dir)

optimizer = optim.Adam(imagen.parameters(), lr=args.OPTIM_LR)

VictoraiTrain(timestamp, args, unets, imagen, train_dataloader, valid_dataloader, training_dir, optimizer, timeout=30)
