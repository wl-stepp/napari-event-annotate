from typing import TYPE_CHECKING
import numpy as np
import tifffile
import napari
import time
import copy
from pathlib import Path

def fluo_graph(viewer, layer: int = 0):
    for frame in viewer.layers[layer].data[::10]:
        if frame.max() > 1:
            print(".", end="")
        else:
            print(" ", end="")
    print(" ")
            
    for frame in viewer.layers[layer].data[::10]:
        if frame.max() > 1:
            print(" ", end="")
        else:
            print(".", end="")
    print(" ")        
    i = 0
    for frame in viewer.layers[layer].data[::10]:
        if i%10 == 0:
            print("\b"*(len(str(i//10)) - 1), end="")
            print(str(i//10), end="")
        else:
            print(" ", end="")
        i +=1

def load_event(viewer: napari.Viewer, folder, ev_n):
    for layer in viewer.layers:
        viewer.layers.clear()
    time.sleep(1)
    events = list(Path(folder).rglob("ev_cos7*"))
    events.sort()
    folder = events[ev_n]
    print(folder.parts[-1])
    viewer.open(f"{folder}/images.tif",)
    viewer.layers["images"].colormap = 'gray_r'
    lower_percentile = 10
    upper_percentile = 90
    viewer.layers["images"].contrast_limits = [np.percentile(viewer.layers["images"].data, lower_percentile),
                                               np.percentile(viewer.layers["images"].data, upper_percentile)]
    try:
        viewer.open(f"{folder}/01images.tif",)
        viewer.layers["01images"].blending = 'additive'
        viewer.layers["01images"].colormap = 'magenta'
        lower_percentile = 10
        upper_percentile = 90
        viewer.layers["images"].contrast_limits = [np.percentile(viewer.layers["01images"].data, lower_percentile),
                                               np.percentile(viewer.layers["01images"].data, upper_percentile)]
    except:
        pass
    try:
        viewer.open(f"{folder}/02images.tif",)
        viewer.layers["02images"].blending = 'additive'
        viewer.layers["02images"].colormap = 'cyan'
        lower_percentile = 10
        upper_percentile = 90
        viewer.layers["images"].contrast_limits = [np.percentile(viewer.layers["02images"].data, lower_percentile),
                                               np.percentile(viewer.layers["02images"].data, upper_percentile)]
    except:
        pass
    viewer.open(f"{folder}/ground_truth.tif",)
    viewer.layers["ground_truth"].colormap = 'inferno'
    viewer.layers["ground_truth"].blending = 'additive'
    
def hide_phase(viewer):
    data = copy.deepcopy(viewer.layers['images'].data)
    viewer.add_image(data, name='phase', colormap='gray')
    viewer.layers.move(len(viewer.layers)-1,0)
    for frame in range(viewer.layers['01images'].data.shape[0]):
        if viewer.layers["01images"].data[frame][0][0] > 0:
            viewer.layers["phase"].data[frame] = np.zeros((viewer.layers["phase"].data.shape[1], viewer.layers["phase"].data.shape[2]))
    viewer.layers["images"].visible = False


def adjust_layers(viewer):
    viewer.layers[3].contrast_limits = [0, 10000]
    viewer.layers[3].colormap = "inferno"
    viewer.layers[1].contrast_limits = [0, 3000]
    viewer.layers[2].contrast_limits = [0, 3000]
    viewer.layers[0].colormap = "gray"


def save_stack(viewer, name: str):
    meta = {'axes': "TCYX", "Composite mode": "composite", "Ranges": (), "LUTs": []}
    stack = np.zeros([len(viewer.layers)] + list(viewer.layers[0].data.shape))
    for idx, layer in enumerate(viewer.layers):
        stack[idx] = layer.data
        meta["Ranges"] = meta["Ranges"] + tuple(layer.contrast_limits[::-1])
        colormap = (np.moveaxis(layer.colormap.colors[:, :-1], 0, 1)*255).astype(np.uint8)
        if colormap.shape[1] < 255:
            new_colormap = np.zeros((3,256))
            for i in range(colormap.shape[0]):
                    new_colormap[i] = np.linspace(colormap[i][0], colormap[i][1], 256).astype(np.uint8)
            colormap = new_colormap.astype(np.uint8)
        meta["LUTs"].append(colormap)
    meta["LUTs"] = meta["LUTs"][::-1]
    meta["Ranges"] = meta["Ranges"][::-1]
    stack = np.flip(stack, 0)
    stack = np.moveaxis(stack, 0, 1)
    stack=stack.astype(np.uint16)
    if not Path(name).parent.exists():
        name = f"/mnt/w/deep_events/experiments/exploration/stacks/{name}.tiff"
    tifffile.imwrite(name, imagej=True, data=stack, metadata=meta)

