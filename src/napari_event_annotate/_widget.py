"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy import QtWidgets, QtCore
import tifffile
import numpy as np
from queue import Queue
from scipy.stats import multivariate_normal
import time

if TYPE_CHECKING:
    import napari



class Editor_Widget(QtWidgets.QWidget):
    """This doct widget is an editor in which, given the layer of previously predicted events, it can edit (add/delete) events

    Parameters
    ----------

    napari_viewer : napari.Viewer
        the viewer that the editor will edit the events from
    """
    def __init__(self, napari_viewer, nbh_size = 10):
        super().__init__()
        self._viewer: napari.Viewer = napari_viewer
        self.setLayout(QtWidgets.QVBoxLayout())
        self.nbh_size = nbh_size
        self.time_data = None
        self.image_path = None

        self.eda_layer = None
        self.eda_ready = False
        self.on_off_score = 0
        self.undo_score = 0
        self.undo_arr = np.zeros((10,2048,2048))

        self.create_EDA_layer_selector()

        self.create_size_slider()

        self.create_top_buttons()

        self.event_list = QtWidgets.QListWidget()

        self.create_bottom_buttons()

        self.layout().addLayout(self.choose_eda_line)
        self.layout().addLayout(self.size_grid)
        self.layout().addLayout(self.top_btn_layout)
        self.layout().addWidget(self.event_list)
        self.layout().addLayout(self.bottom_btn_layout)

        if len(self._viewer.layers) > 0:
            self.init_data()

        self.Twait = 2500
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.Twait)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.init_data)

      #events
        self._viewer.layers.events.inserted.connect(self.init_after_timer)
        #self.edit_btn.clicked.connect(self.get_coordinates)
        self._viewer.layers.events.removed.connect(self.eliminate_widget_if_empty)

        self._viewer.layers.events.inserted.connect(self.update_eda_layer_chooser)
        self._viewer.layers.events.removed.connect(self.update_eda_layer_chooser)
        #self._viewer.mouse_drag_callbacks.append(self.get_coordinates)

        @self._viewer.mouse_drag_callbacks.append
        def get_event(viewer, event):
            """Distiguishes between a click and a drag. Click if below a certain time (80 ms)"""
            if event.type == 'mouse_press':
                self.click_time = time.perf_counter()
                print('mouse down')
                dragged = False
                yield
            # on move
            while event.type == 'mouse_move':
                drag_time = time.perf_counter() - self.click_time
                print(drag_time)
                if drag_time > 0.08:
                    dragged = True
                yield
            if dragged:
                print('drag end')
            else:
                print('clicked!')
                self.get_coordinates(event.position)

    # Functions for the GUI creation
    def hideEvent(self, event):
        self._viewer: napari.Viewer = None
        self.nbh_size = None
        self.time_data = None
        self.image_path = None

        self.eda_layer = None
        self.eda_ready = False
        self.on_off_score = 0
        self.undo_score = 0
        event.accept()
        print('Editor Widget is closed now')


    def create_EDA_layer_selector(self):
        """Creates the selector for the EDA layer"""
        self.choose_eda_line = QtWidgets.QHBoxLayout()
        self.choose_eda_line.addWidget(QtWidgets.QLabel('NN Images layer'))
        self.eda_layer_chooser = QtWidgets.QComboBox()
        for lay in self._viewer.layers:
            self.eda_layer_chooser.addItem(lay.name)
        self.choose_eda_line.addWidget(self.eda_layer_chooser)

    def create_size_slider(self):
        self.size_grid = QtWidgets.QGridLayout()
        self.size_slider= QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setSingleStep(1)
        self.size_slider.setMaximum(10)
        self.size_slider.setMinimumWidth(150)
        self.size_grid.addWidget(QtWidgets.QLabel('Gaussian Size'),0,0)
        self.size_show = QtWidgets.QLabel('-')
        self.size_grid.addWidget(self.size_show,0,1)
        self.size_grid.addWidget(self.size_slider,1,0,1,2)

    def create_top_buttons(self):
        self.edit_btn = QtWidgets.QPushButton('Edit')
        self.undo_btn = QtWidgets.QPushButton('Undo')
        self.edit_btn.setStyleSheet(""" background-color: "None"; """)
        self.undo_btn.setStyleSheet(""" QPushButton {background-color: "None";} QPushButton:pressed { background-color: "darkGray";} """)
        self.top_btn_layout = QtWidgets.QHBoxLayout()
        self.top_btn_layout.addWidget(self.edit_btn)
        self.top_btn_layout.addWidget(self.undo_btn)
        self.edit_btn.clicked.connect(self.on_off)
        self.undo_btn.clicked.connect(self.undo)

    def create_bottom_buttons(self):
        self.save_all_btn = QtWidgets.QPushButton('Save Image')
        self.bottom_btn_layout = QtWidgets.QHBoxLayout()
        self.bottom_btn_layout.addWidget(self.save_all_btn)
        self.save_all_btn.clicked.connect(self.save_all_events)

    ##### BUTTON HAS ON-OFF STATES WITH DIFFERENT ON_OFF_SCORE #####
    def on_off(self):
        if self.on_off_score == 0:
            self.on_off_score = self.on_off_score+1
            self.edit_btn.setStyleSheet(""" background-color: "darkGray"; """)
            gauss=self.get_gaussian(2,3,[0, 0])
        elif self.on_off_score == 1:
            self.on_off_score = self.on_off_score-1
            self.edit_btn.setStyleSheet(""" background-color: "None"; """)
        # self.edit_btn.setCheckable(True);
        # self.edit_btn.setStyleSheet(QString("QPushButton {background-color: gray;}"));

    ##### EDA LAYER: CHOOSING LAYER ---> EDA_LAYER UPDATES #####
    def update_eda_layer_from_chooser(self, text = None):
        if text is None:
            self.search_eda_layer()
            text = self.eda_layer_chooser.currentText()
        if text != '':
            self.eda_layer = self._viewer.layers[text]
            self.size_slider.setValue(5)
            self.eda_ready = True

    ##### EDA LAYER: IF LAYER IS ADDED OR REMOVED ---> CHOOSER OPTIONS UPDATE #####
    def update_eda_layer_chooser(self):
        self.eda_layer_chooser.clear()
        for lay in self._viewer.layers:
            self.eda_layer_chooser.addItem(lay.name)

    ##### EDA LAYER: LOOK THROUGH AVAILABLE LAYERS AND CHECK IF A LAYER IS CALLED 'NN IMAGES' ---> EDA_LAYER BECOMES 'NN IMAGES' #####
    def search_eda_layer(self):
        self.eda_ready = False
        for lay in self._viewer.layers:
            if lay.name == 'NN Images':
                self.eda_layer = lay
                self.eda_ready = True
                try:
                    self.eda_layer_chooser.setCurrentText('NN Images')
                except:
                    print('No compatible layer in the selector')
        if not self.eda_ready:
            self._viewer.add_image(np.zeros(self._viewer.layers[0].data.shape), name="NN Images", blending="additive",
                                   scale=self._viewer.layers[0].scale, colormap='red')
            self.update_eda_layer_chooser()
            self.update_eda_layer_from_chooser()

    ##### SAVE #####
    def save_all_events(self):
        for lay in self._viewer.layers:
            data=self.eda_layer.data
            currname= f'{lay.name}_edit.tiff'
            directory = r'C:\Users\roumba\Documents\Software\deep-events'
            savepath = directory + f'\{currname}'
            tifffile.imwrite(savepath, (data).astype(np.uint64), photometric='minisblack')


    def get_gaussian(self, sigma, sz, offset):
        size = (sz+1, sz+1)
        mu = ((sz+1)/2 + offset[0], (sz+1)/2 + offset[1])
        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        cov = ((sigma/2)**2) * np.eye(2)
        mvn = multivariate_normal(mean=mu, cov=cov)
        gauss = mvn.pdf(pos)
        return gauss

    ##### ADD #####
    def add_gauss(self,sz, offset):
        """ Function that adds Gaussian intensity from NN Images """
        sigma = self.size_slider.value()
        gaussian_points = self.get_gaussian(sigma, sz, offset)                                                        #convers tensor into numpy array
        gaussian_points = gaussian_points/np.max(gaussian_points)                                               #divides by the max
        gaussian_points[gaussian_points < 0.1] = 0                                                              #sets background to zero
        gaussian_points = gaussian_points/np.max(gaussian_points)                                               #divides by max again
        return gaussian_points

    ##### REMOVE SQUARES #####
    def remove_int(self,mu,fr_num):
        """ Function that removes intensity from NN Images """
        intensity =  self.eda_layer.data[fr_num, mu[0], mu[1]]
        tolerance = intensity*0.99
        self.eda_layer.data[fr_num, :, :] = flood_fill(self.eda_layer.data[fr_num], (mu[0], mu[1]))                                                                                        #divides by max again
        #self.eda_layer.data[fr_num, ymin:ymax, xmin:xmax] = 0
        new_data = self.eda_layer.data
        return new_data

        ##### UNDO BUTTON #####
    def undo(self):
        frame_num = int(getattr(self.eda_layer, 'position')[0])
        if self.undo_score != 0:
            self.undo_score = self.undo_score - 1
            self.eda_layer.data[frame_num] = self.undo_arr[self.undo_score]
            self.eda_layer.refresh()
        else:
            print('Undo unavailable')

    def get_coordinates(self, data_coordinates):
        data_coordinates = [data/scale for data, scale in zip(data_coordinates, self._viewer.layers[0].scale)]
        if self.on_off_score==1:
            frame_num = round(data_coordinates[0])
            # print("DATA COORDS", data_coordinates)
            ycoord = data_coordinates[1]
            xcoord = data_coordinates[2]
            self.undo_arr[self.undo_score]= self.eda_layer.data[frame_num]
            mu = (ycoord,xcoord)
            sigma = self.size_slider.value()
            size = np.ceil(sigma * 2.5)
            pixel_coords = [round(x) for x in data_coordinates]
            offset = [x - y for x, y in zip(data_coordinates[1:], pixel_coords[1:])]
            # print("PIXEL COORDS", pixel_coords)
            int_val = self.eda_layer.data[pixel_coords[0],pixel_coords[1],pixel_coords[2]]
            if  int_val < 0.1:
                xmax= int(np.ceil(pixel_coords[2]+(size/2)))
                xmin= int(np.floor(pixel_coords[2]-(size/2)))
                ymax= int(np.ceil(pixel_coords[1]+(size/2)))
                ymin= int(np.floor(pixel_coords[1]-(size/2)))
                print('Intensity Value is', int_val)
                self.eda_layer.data[frame_num, ymin:ymax, xmin:xmax] = self.eda_layer.data[frame_num, ymin:ymax, xmin:xmax] + self.add_gauss(size, offset)
                self.undo_score=self.undo_score+1
                if self.undo_score==10:
                    for i in range(1,10):
                        self.undo_arr[i-1]=self.undo_arr[i]
                    self.undo_score=9
            else:
                print('Intensity Value is', int_val)
                mu = [round(x) for x in mu]
                self.eda_layer.data = self.remove_int(mu,frame_num)
                self.undo_score=self.undo_score+1
                if self.undo_score==10:
                    for i in range(1,10):
                        self.undo_arr[i-1]=self.undo_arr[i]
                    self.undo_score=9
            self.eda_layer.refresh()

    ##### SLIDER VALUE SHOWN #####
    def update_size(self):
        self.size= self.size_slider.value()
        self.size_show.setText(str(self.size))


    ##### INITIALIZING THE DATA #####
    def init_data(self):
        """Initialize data from the layers"""

        self.update_eda_layer_chooser()
        self.eda_layer_chooser.currentTextChanged.connect(self.update_eda_layer_from_chooser)
        self.update_eda_layer_from_chooser()
        self.size_slider.valueChanged.connect(self.update_size)
        self.size_slider.setValue(5)
        if self.eda_ready:
            self.update_size()
        self.undo_arr = np.zeros([10] + list(self._viewer.layers[self.eda_layer_chooser.currentText()].data.shape[1:]))


    def eliminate_widget_if_empty(self,event):
        if len(event.source)==0:
            try:
                self._viewer.window.remove_dock_widget(self)
                self.image_path=None
            except:
                print('Dock already deleted')

    def init_after_timer(self): ##wooow directly put in connect
        if len(self._viewer.layers) < 2:
            self.timer.start(self.Twait) #restarts the timer with a timeout of Twait ms



def flood_fill(img, seed):
    """ Special flood fill to accept all values down to a specific value """
    height, width = img.shape
    filled = np.zeros((height, width), dtype=bool)
    q = Queue()
    q.put(seed)

    while not q.empty():
        x, y = q.get()

        if filled[x, y]:
            continue

        if np.abs(img[x, y]) == 0:
            continue

        img[x, y] = 0
        filled[x, y] = True

        if x > 0:
            q.put((x - 1, y))
        if x < height - 1:
            q.put((x + 1, y))
        if y > 0:
            q.put((x, y - 1))
        if y < width - 1:
            q.put((x, y + 1))

    return img