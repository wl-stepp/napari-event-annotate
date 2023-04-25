import sys
sys.path.append("./src/napari_event_annotate")
import napari
import os


viewer = napari.Viewer()
#path=  str(os.path.dirname(__file__))+'/images/steven_14_MMStack_Injection.ome.tif'                 #/steven_192.ome.zarr/Images'#"https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/9836842.zarr/"
#savepath = str(os.path.dirname(__file__))+'/images/steven_192.ome.zarr/Reels/a.eda'
#path= str(os.path.dirname(__file__))+'/images/example_image.tif'
#viewer.open(path, plugin='aicsimageio-in-memory')#, plugin = 'napari-ome-zarr')
viewer.window.add_plugin_dock_widget('napari-event-annotate','Cropper') #'Add time scroller'
path= str(os.path.dirname(__file__))+'/test_data/cos7_isim_mtstaygold_fluo.ome.tif'
viewer.open(path, plugin='napari-aicsimageio')
#viewer.layers.save(savepath, plugin='napari-eda-highlight-reel')
napari.run()