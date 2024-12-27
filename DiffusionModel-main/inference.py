from argparse import ArgumentParser
from modelfiles.generate import load_victorai, sample_and_save

parser = ArgumentParser()
parser.add_argument("-c", "--CAPTIONS", dest="CAPTIONS", help="Single caption to generate for or filepath for .txt "
                                                              "file of captions to generate for", default=None, type=str)
parser.add_argument("-d", "--TRAINING_DIRECTORY", dest="TRAINING_DIRECTORY", help="Training directory to use for inference", type=str)
args = parser.parse_args()

victorai = load_victorai(args.TRAINING_DIRECTORY)

if args.CAPTIONS is None:
    print("\nNo caption supplied - using the default of \"a happy dog\".\n")
    captions = ['a happy dog']
elif not args.CAPTIONS.endswith(".txt"):
    captions = [args.CAPTIONS]
elif args.CAPTIONS.endswith(".txt"):
    with open(args.CAPTIONS, 'r') as f:
        lines = f.readlines()
    captions = [line[:-1] if line.endswith('\n') else line for line in lines]
else:
    raise ValueError("Please input a valid argument for --CAPTIONS")


sample_and_save(captions, training_directory=args.TRAINING_DIRECTORY, sample_args={'cond_scale':3.})

