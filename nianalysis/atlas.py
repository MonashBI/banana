from arcana.data.collection import FilesetCollection
from arcana.node import Node
from .requirement import fsl5_req


req = fsl5_req.best_requirement(fsl5_req, Node.available_modules())
Node.load_module(req.name, req.version)
fsl_path = 
