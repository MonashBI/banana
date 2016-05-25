#!/usr/bin/env python
"""
Plots a workflow graph using NiPype's write_graph function and then loads and
plots the resulting PNG
"""
import os
from argparse import ArgumentParser
import matplotlib.image as img
import matplotlib.pyplot as plt
from neuroanalysis.mri import DiffusionDataset

parser = ArgumentParser()
parser.add_argument('workflow', type=str,
                    help="The workflow to plot, can be one of: 'diffusion'")
parser.add_argument('--style', type=str, default='flat',
                    help=("The style of the graph, can be one of can be one of"
                          " 'orig', 'flat', 'exec', 'hierarchical'"))
parser.add_argument('--detailed', action='store_true', default=False,
                    help="Plots a detailed version of the graph")
parser.add_argument('--work_dir', default=None, type=str,
                    help="The work directory where the graphs will be created")
parser.add_argument('--save', type=str, default=None,
                    help=("Save the created PNG to the given filename"))
args = parser.parse_args()

if args.work_dir is not None:
    os.chdir(args.work_dir)
# Create workflow
workflow, _, _ = getattr(DiffusionDataset, args.workflow + '_workflow')()
# Write workflow
workflow.write_graph(graph2use=args.style)
# Plot worfklow
if args.detailed:
    graph_file = 'graph_detailed.dot.png'
else:
    graph_file = 'graph.dot.png'
graph = img.imread(graph_file)
plt.imshow(graph)
# Clean up created graph files
if args.save is not None:
    os.rename(graph_file, args.save)
# Clean up graph files
os.remove('graph.dot')
os.remove('graph_detailed.dot')
try:
    os.remove('graph.dot.png')
except OSError:
    pass
try:
    os.remove('graph_detailed.dot.png')
except OSError:
    pass
# Show graph
plt.show()
