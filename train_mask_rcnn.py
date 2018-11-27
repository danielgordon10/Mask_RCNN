import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import constants
if constants.EQA:
    from dataset_readers.eqa_mask_rcnn import eqa_dataset as dataset
else:
    from dataset_readers.ai2thor_mask_rcnn import ai2thor_dataset as dataset
dataset.main(sys.argv[1:])
