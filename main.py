# This is the main function file
import klayout.db as db
from utils import *
import os

# Global configuration
# A predetermined dict, indicating the original layout to be manufactured.
# Used by layout.find_layer(target_layer, 0)
# Return a list, because there is may be more than 1 layer to be adjusted.
TARGET_LAYERS = {'demo2_65nm': [13], 'demo3_162nm': [1], 'inv':[46], 'Case':[1], "xor":[46], "nand2":[46]}

GDS_NAME = 'xor'
INPUT_GDS_FILE = GDS_NAME + '.gds'
OUTPUT_GDS_FILE = GDS_NAME + '_mod.gds2'

def main():
    if not os.path.exists(INPUT_GDS_FILE):
        prepare_data(INPUT_GDS_FILE)
        print(INPUT_GDS_FILE + " is generated.")
    else:
        print(INPUT_GDS_FILE + " is prepared.")

    layout = db.Layout()
    layout.read(INPUT_GDS_FILE)
    # print(layout.dbu)
    for cell_index in layout.each_cell_top_down():
        print("Processing cell", cell_index, layout.cell(cell_index).name)
        for target_layer in TARGET_LAYERS[GDS_NAME]:
            layer_index = layout.find_layer(target_layer, 0)
            process_layer(layout.cell(cell_index), layer_index)

    layout.write(OUTPUT_GDS_FILE)
    print(OUTPUT_GDS_FILE + " is generated.")

if __name__ == '__main__':
    main()
