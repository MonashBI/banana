from banana.analysis.mri import DwiAnalysis
from argparse import ArgumentParser
import os


parser = ArgumentParser()
parser.add_argument('input_image', type=str,
                    help="Input dwi image in mif format (with gradients)")
parser.add_argument('output_dir', type=str,
                    help=("Output directory where FA, CSD and tracks files"
                          " are stored."))
parser.add_argument('--working_dir', type=str, default=None,
                    help=("The directory where the intermediate files are "
                          "stored"))
args = parser.parse_args()
processor = DwiAnalysis('tclose', os.environ['DARIS_PASSWORD'])
processor.process(args.input_image, args.output_dir,
                  working_dir=args.working_dir)
