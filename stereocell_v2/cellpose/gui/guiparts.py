"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import PyQt5
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QRadioButton, QWidget, QDialog, QButtonGroup, QSlider, QStyle, QStyleOptionSlider, QGridLayout, QPushButton, QLabel, QLineEdit, QDialogButtonBox, QComboBox, QCheckBox
import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph import Point
import numpy as np
import pathlib, os

def create_channel_choose():
    # choose channel
    ChannelChoose = [QComboBox(), QComboBox()]
    ChannelLabels = []
    ChannelChoose[0].addItems(['gray','red','green','blue'])
    ChannelChoose[1].addItems(['none','red','green','blue'])
    cstr = ['chan to segment:', 'chan2 (optional): ']
    for i in range(2):
        ChannelLabels.append(QLabel(cstr[i]))
        if i==0:
            ChannelLabels[i].setToolTip('this is the channel in which the cytoplasm or nuclei exist that you want to segment')
            ChannelChoose[i].setToolTip('this is the channel in which the cytoplasm or nuclei exist that you want to segment')
        else:
            ChannelLabels[i].setToolTip('if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option')
            ChannelChoose[i].setToolTip('if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option')
        
    return ChannelChoose, ChannelLabels

class ModelButton(QPushButton):
    def __init__(self, parent, model_name, text):
        super().__init__()
        self.setEnabled(False)
        self.setStyleSheet(parent.styleInactive)
        self.setText(text)
        self.setFont(parent.smallfont)
        self.clicked.connect(lambda: self.press(parent))
        self.model_name = model_name
        
    def press(self, parent):
        for i in range(len(parent.StyleButtons)):
            parent.StyleButtons[i].setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        parent.compute_model(self.model_name)

class TrainWindow(QDialog):
    def __init__(self, parent, model_strings):
        super().__init__(parent)
        self.setGeometry(100,100,900,350)
        self.setWindowTitle('train settings')
        self.win = QWidget(self)
        self.l0 = QGridLayout()
        self.win.setLayout(self.l0)

        yoff = 0
        qlabel = QLabel('train model w/ images + _seg.npy in current folder >>')
        qlabel.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        
        qlabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, yoff,0,1,2)

        # choose initial model
        yoff+=1
        self.ModelChoose = QComboBox()
        self.ModelChoose.addItems(model_strings)
        self.ModelChoose.addItems(['scratch']) 
        self.ModelChoose.setFixedWidth(150)
        self.ModelChoose.setCurrentIndex(parent.training_params['model_index'])
        self.l0.addWidget(self.ModelChoose, yoff, 1,1,1)
        qlabel = QLabel('initial model: ')
        qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, yoff,0,1,1)

        # choose channels
        self.ChannelChoose, self.ChannelLabels = create_channel_choose()
        for i in range(2):
            yoff+=1
            self.ChannelChoose[i].setFixedWidth(150)
            self.ChannelChoose[i].setCurrentIndex(parent.ChannelChoose[i].currentIndex())
            self.l0.addWidget(self.ChannelLabels[i], yoff, 0,1,1)
            self.l0.addWidget(self.ChannelChoose[i], yoff, 1,1,1)

        # choose parameters        
        labels = ['learning_rate', 'weight_decay', 'n_epochs', 'model_name']
        self.edits = []
        yoff += 1
        for i, label in enumerate(labels):
            qlabel = QLabel(label)
            qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.l0.addWidget(qlabel, i+yoff,0,1,1)
            self.edits.append(QLineEdit())
            self.edits[-1].setText(str(parent.training_params[label]))
            self.edits[-1].setFixedWidth(200)
            self.l0.addWidget(self.edits[-1], i+yoff, 1,1,1)

        yoff+=len(labels)

        yoff+=1
        qlabel = QLabel('(to remove files, click cancel then remove \nfrom folder and reopen train window)')
        self.l0.addWidget(qlabel, yoff,0,2,4)

        # click button
        yoff+=2
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(lambda: self.accept(parent))
        self.buttonBox.rejected.connect(self.reject)
        self.l0.addWidget(self.buttonBox, yoff, 0, 1,4)

        
        # list files in folder
        qlabel = QLabel('filenames')
        qlabel.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.l0.addWidget(qlabel, 0,4,1,1)
        qlabel = QLabel('# of masks')
        qlabel.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.l0.addWidget(qlabel, 0,5,1,1)
    
        for i in range(10):
            if i > len(parent.train_files) - 1:
                break
            elif i==9 and len(parent.train_files) > 10:
                label = '...'
                nmasks = '...'
            else:
                label = os.path.split(parent.train_files[i])[-1]
                nmasks = str(parent.train_labels[i].max())
            qlabel = QLabel(label)
            self.l0.addWidget(qlabel, i+1,4,1,1)
            qlabel = QLabel(nmasks)
            qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.l0.addWidget(qlabel, i+1, 5,1,1)

    def accept(self, parent):
        # set channels
        for i in range(2):
            parent.ChannelChoose[i].setCurrentIndex(self.ChannelChoose[i].currentIndex())
        # set training params
        parent.training_params = {'model_index': self.ModelChoose.currentIndex(),
                                 'learning_rate': float(self.edits[0].text()), 
                                 'weight_decay': float(self.edits[1].text()), 
                                 'n_epochs':  int(self.edits[2].text()),
                                 'model_name': self.edits[3].text()
                                 }
        self.done(1)
        
def make_quadrants(parent, yp):
    """ make quadrant buttons """
    parent.quadbtns = QButtonGroup(parent)
    for b in range(9):
        btn = QuadButton(b, ' '+str(b+1), parent)
        parent.quadbtns.addButton(btn, b)
        parent.l0.addWidget(btn, yp + parent.quadbtns.button(b).ypos, 5+parent.quadbtns.button(b).xpos, 1, 1)
        btn.setEnabled(True)
        b += 1
    parent.quadbtns.setExclusive(True)

class QuadButton(QPushButton):
    """ custom QPushButton class for quadrant plotting
        requires buttons to put into a QButtonGroup (parent.quadbtns)
         allows only 1 button to pressed at a time
    """
    def __init__(self, bid, Text, parent=None):
        super(QuadButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleUnpressed)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.setMaximumWidth(22)
        self.xpos = bid%3
        self.ypos = int(np.floor(bid/3))
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()

    def press(self, parent, bid):
        for b in range(9):
            if parent.quadbtns.button(b).isEnabled():
                parent.quadbtns.button(b).setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        self.xrange = np.array([self.xpos-.2, self.xpos+1.2]) * parent.Lx/3
        self.yrange = np.array([self.ypos-.2, self.ypos+1.2]) * parent.Ly/3
        # change the zoom
        parent.p0.setXRange(self.xrange[0], self.xrange[1])
        parent.p0.setYRange(self.yrange[0], self.yrange[1])
        parent.show()

def horizontal_slider_style():
    return """QSlider::groove:horizontal {
            border: 1px solid #bbb;
            background: black;
            height: 10px;
            border-radius: 4px;
            }

            QSlider::sub-page:horizontal {
            background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,
                stop: 0 black, stop: 1 rgb(150,255,150));
            background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,
                stop: 0 black, stop: 1 rgb(150,255,150));
            border: 1px solid #777;
            height: 10px;
            border-radius: 4px;
            }

            QSlider::add-page:horizontal {
            background: black;
            border: 1px solid #777;
            height: 10px;
            border-radius: 4px;
            }

            QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #eee, stop:1 #ccc);
            border: 1px solid #777;
            width: 13px;
            margin-top: -2px;
            margin-bottom: -2px;
            border-radius: 4px;
            }

            QSlider::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #fff, stop:1 #ddd);
            border: 1px solid #444;
            border-radius: 4px;
            }

            QSlider::sub-page:horizontal:disabled {
            background: #bbb;
            border-color: #999;
            }

            QSlider::add-page:horizontal:disabled {
            background: #eee;
            border-color: #999;
            }

            QSlider::handle:horizontal:disabled {
            background: #eee;
            border: 1px solid #aaa;
            border-radius: 4px;
            }"""

class ExampleGUI(QDialog):
    def __init__(self, parent=None):
        super(ExampleGUI, self).__init__(parent)
        self.setGeometry(100,100,1300,900)
        self.setWindowTitle('GUI layout')
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        guip_path = pathlib.Path.home().joinpath('.cellpose', 'cellpose_gui.png')
        guip_path = str(guip_path.resolve())
        pixmap = QPixmap(guip_path)
        label = QLabel(self)
        label.setPixmap(pixmap)
        pixmap.scaled
        layout.addWidget(label, 0, 0, 1, 1)

class HelpWindow(QDialog):
    def __init__(self, parent=None):
        super(HelpWindow, self).__init__(parent)
        self.setGeometry(100,50,700,850)
        self.setWindowTitle('cellpose help')
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        
        text = ('''
            <p class="has-line-data" data-line-start="5" data-line-end="6">Main GUI mouse controls:</p>
            <ul>
            <li class="has-line-data" data-line-start="7" data-line-end="8">Pan  = left-click  + drag</li>
            <li class="has-line-data" data-line-start="8" data-line-end="9">Zoom = scroll wheel (or +/= and - buttons) </li>
            <li class="has-line-data" data-line-start="9" data-line-end="10">Full view = double left-click</li>
            <li class="has-line-data" data-line-start="10" data-line-end="11">Select mask = left-click on mask</li>
            <li class="has-line-data" data-line-start="11" data-line-end="12">Delete mask = Ctrl (or COMMAND on Mac) + left-click</li>
            <li class="has-line-data" data-line-start="11" data-line-end="12">Merge masks = Alt + left-click (will merge last two)</li>
            <li class="has-line-data" data-line-start="12" data-line-end="13">Start draw mask = right-click</li>
            <li class="has-line-data" data-line-start="13" data-line-end="15">End draw mask = right-click, or return to circle at beginning</li>
            </ul>
            <p class="has-line-data" data-line-start="15" data-line-end="16">Overlaps in masks are NOT allowed. If you \
            draw a mask on top of another mask, it is cropped so that it doesn’t overlap with the old mask. Masks in 2D \
            should be single strokes (single stroke is checked). If you want to draw masks in 3D (experimental), then \
            you can turn this option off and draw a stroke on each plane with the cell and then press ENTER. 3D \
            labelling will fill in planes that you have not labelled so that you do not have to as densely label.</p>
            <p class="has-line-data" data-line-start="17" data-line-end="18">!NOTE!: The GUI automatically saves after \
            you draw a mask in 2D but NOT after 3D mask drawing and NOT after segmentation. Save in the file menu or \
            with Ctrl+S. The output file is in the same folder as the loaded image with <code>_seg.npy</code> appended.</p>
            <table class="table table-striped table-bordered">
            <br><br>
            FYI there are tooltips throughout the GUI (hover over text to see)
            <br>
            <thead>
            <tr>
            <th>Keyboard shortcuts</th>
            <th>Description</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>=/+  button // - button</td>
            <td>zoom in // zoom out</td>
            </tr>
            <tr>
            <td>CTRL+Z</td>
            <td>undo previously drawn mask/stroke</td>
            </tr>
            <tr>
            <td>CTRL+Y</td>
            <td>undo remove mask</td>
            </tr>
            <tr>
            <td>CTRL+0</td>
            <td>clear all masks</td>
            </tr>
            <tr>
            <td>CTRL+L</td>
            <td>load image (can alternatively drag and drop image)</td>
            </tr>
            <tr>
            <td>CTRL+S</td>
            <td>SAVE MASKS IN IMAGE to <code>_seg.npy</code> file</td>
            </tr>
            <tr>
            <td>CTRL+T</td>
            <td>train model using _seg.npy files in folder
            </tr>
            <tr>
            <td>CTRL+P</td>
            <td>load <code>_seg.npy</code> file (note: it will load automatically with image if it exists)</td>
            </tr>
            <tr>
            <td>CTRL+M</td>
            <td>load masks file (must be same size as image with 0 for NO mask, and 1,2,3… for masks)</td>
            </tr>
            <tr>
            <td>CTRL+N</td>
            <td>save masks as PNG</td>
            </tr>
            <tr>
            <td>CTRL+R</td>
            <td>save ROIs to native ImageJ ROI format</td>
            </tr>
            <tr>
            <td>CTRL+F</td>
            <td>save flows to image file</td>
            </tr>
            <tr>
            <td>A/D or LEFT/RIGHT</td>
            <td>cycle through images in current directory</td>
            </tr>
            <tr>
            <td>W/S or UP/DOWN</td>
            <td>change color (RGB/gray/red/green/blue)</td>
            </tr>
            <tr>
            <td>R / G / B </td>
            <td>toggle between RGB and Red or Green or Blue</td>
            </tr>
            <tr>
            <td>PAGE-UP / PAGE-DOWN</td>
            <td>change to flows and cell prob views (if segmentation computed)</td>
            </tr>
            <tr>
            <td>X</td>
            <td>turn masks ON or OFF</td>
            </tr>
            <tr>
            <td>Z</td>
            <td>toggle outlines ON or OFF</td>
            </tr>
            <tr>
            <td>, / .</td>
            <td>increase / decrease brush size for drawing masks</td>
            </tr>
            </tbody>
            </table>
            <p class="has-line-data" data-line-start="36" data-line-end="37"><strong>Segmentation options (2D only) </strong></p>
            <p class="has-line-data" data-line-start="38" data-line-end="39">SIZE: you can manually enter the approximate diameter for your cells, or press “calibrate” to let the model estimate it. The size is represented by a disk at the bottom of the view window (can turn this disk of by unchecking “scale disk on”).</p>
            <p class="has-line-data" data-line-start="40" data-line-end="41">use GPU: if you have specially installed the cuda version of mxnet, then you can activate this, but it won’t give huge speedups when running single 2D images in the GUI.</p>
            <p class="has-line-data" data-line-start="42" data-line-end="43">MODEL: there is a <em>cytoplasm</em> model and a <em>nuclei</em> model, choose what you want to segment</p>
            <p class="has-line-data" data-line-start="44" data-line-end="45">CHAN TO SEG: this is the channel in which the cytoplasm or nuclei exist</p>
            <p class="has-line-data" data-line-start="46" data-line-end="47">CHAN2 (OPT): if <em>cytoplasm</em> model is chosen, then choose the nuclear channel for this option</p>
            ''')
        label = QLabel(text)
        label.setFont(QtGui.QFont("Arial", 8))
        label.setWordWrap(True)
        layout.addWidget(label, 0, 0, 1, 1)
        self.show()


class TrainHelpWindow(QDialog):
    def __init__(self, parent=None):
        super(TrainHelpWindow, self).__init__(parent)
        self.setGeometry(100,50,700,300)
        self.setWindowTitle('training instructions')
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        
        text = ('''
            Check out this <a href="https://youtu.be/3Y1VKcxjNy4">video</a> to learn the process.
            <ol>
                <li>Drag and drop an image from a folder of images with a similar style (like similar cell types).</li>
                <li>Run the built-in models on one of the images using the "model zoo" and find the one that works best for your data. Make sure that if you have a nuclear channel you have selected it for CHAN2.</li>
                <li>Fix the labelling by drawing new ROIs (right-click) and deleting incorrect ones (CTRL+click). The GUI autosaves any manual changes (but does not autosave after running the model, for that click CTRL+S). The segmentation is saved in a "_seg.npy" file.</li>
                <li> Go to the "Models" menu in the File bar at the top and click "Train new model..." or use shortcut CTRL+T. </li>
                <li> Choose the pretrained model to start the training from (the model you used in #2), and type in the model name that you want to use. The other parameters should work well in general for most data types. Then click OK. </li>
                <li> The model will train (much faster if you have a GPU) and then auto-run on the next image in the folder. Next you can repeat #3-#5 as many times as is necessary. </li>
                <li> The trained model is available to use in the future in the GUI in the "custom model" section and is saved in your image folder. </li>
            </ol>
            ''')
        label = QLabel(text)
        label.setFont(QtGui.QFont("Arial", 8))
        label.setWordWrap(True)
        layout.addWidget(label, 0, 0, 1, 1)
        self.show()


class TypeRadioButtons(QButtonGroup):
    def __init__(self, parent=None, row=0, col=0):
        super(TypeRadioButtons, self).__init__()
        parent.color = 0
        self.parent = parent
        self.bstr = self.parent.cell_types
        for b in range(len(self.bstr)):
            button = QRadioButton(self.bstr[b])
            button.setStyleSheet('color: rgb(190,190,190);')
            button.setFont(QtGui.QFont("Arial", 10))
            if b==0:
                button.setChecked(True)
            self.addButton(button, b)
            button.toggled.connect(lambda: self.btnpress(parent))
            self.parent.l0.addWidget(button, row+b,col,1,2)
        self.setExclusive(True)
        #self.buttons.

    def btnpress(self, parent):
       b = self.checkedId()
       self.parent.cell_type = b

class RGBRadioButtons(QButtonGroup):
    def __init__(self, parent=None, row=0, col=0):
        super(RGBRadioButtons, self).__init__()
        parent.color = 0
        self.parent = parent
        self.bstr = ["image", "gradXY", "cellprob", "gradZ"]
        #self.buttons = QButtonGroup()
        self.dropdown = []
        for b in range(len(self.bstr)):
            button = QRadioButton(self.bstr[b])
            button.setStyleSheet('color: white;')
            button.setFont(QtGui.QFont("Arial", 10))
            if b==0:
                button.setChecked(True)
            self.addButton(button, b)
            button.toggled.connect(lambda: self.btnpress(parent))
            self.parent.l0.addWidget(button, row,col+2*b,1,2)
        self.setExclusive(True)
        #self.buttons.

    def btnpress(self, parent):
       b = self.checkedId()
       self.parent.view = b
       if self.parent.loaded:
           self.parent.update_plot()


class ViewBoxNoRightDrag(pg.ViewBox):
    def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True, invertY=False, enableMenu=True, name=None, invertX=False):
        pg.ViewBox.__init__(self, None, border, lockAspect, enableMouse,
                            invertY, enableMenu, name, invertX)
        self.parent = parent
        self.axHistoryPointer = -1

    def keyPressEvent(self, ev):
        """
        This routine should capture key presses in the current view box.
        The following events are implemented:
        +/= : moves forward in the zooming stack (if it exists)
        - : moves backward in the zooming stack (if it exists)

        """
        ev.accept()
        if ev.text() == '-':
            self.scaleBy([1.1, 1.1])
        elif ev.text() in ['+', '=']:
            self.scaleBy([0.9, 0.9])
        else:
            ev.ignore()

class ImageDraw(pg.ImageItem):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    GraphicsObject displaying an image. Optimized for rapid update (ie video display).
    This item displays either a 2D numpy array (height, width) or
    a 3D array (height, width, RGBa). This array is optionally scaled (see
    :func:`setLevels <pyqtgraph.ImageItem.setLevels>`) and/or colored
    with a lookup table (see :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`)
    before being displayed.
    ImageItem is frequently used in conjunction with
    :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>` or
    :class:`HistogramLUTWidget <pyqtgraph.HistogramLUTWidget>` to provide a GUI
    for controlling the levels and lookup table used to display the image.
    """

    sigImageChanged = QtCore.pyqtSignal()

    def __init__(self, image=None, viewbox=None, parent=None, **kargs):
        super(ImageDraw, self).__init__()
        #self.image=None
        #self.viewbox=viewbox
        self.levels = np.array([0,255])
        self.lut = None
        self.autoDownsample = False
        self.axisOrder = 'row-major'
        self.removable = False

        self.parent = parent
        #kernel[1,1] = 1
        self.setDrawKernel(kernel_size=self.parent.brush_size)
        self.parent.current_stroke = []
        self.parent.in_stroke = False

    def mouseClickEvent(self, ev):
        if self.parent.masksOn or self.parent.outlinesOn:
            if  self.parent.loaded and (ev.button() == QtCore.Qt.RightButton or 
                    ev.modifiers() & QtCore.Qt.ShiftModifier and not ev.double()):
                if not self.parent.in_stroke:
                    ev.accept()
                    self.create_start(ev.pos())
                    self.parent.stroke_appended = False
                    self.parent.in_stroke = True
                    self.drawAt(ev.pos(), ev)
                else:
                    ev.accept()
                    self.end_stroke()
                    self.parent.in_stroke = False
            elif not self.parent.in_stroke:
                y,x = int(ev.pos().y()), int(ev.pos().x())
                if y>=0 and y<self.parent.Ly and x>=0 and x<self.parent.Lx:
                    if ev.button() == QtCore.Qt.LeftButton and not ev.double():
                        idx = self.parent.cellpix[self.parent.currentZ][y,x]
                        if idx > 0:
                            if ev.modifiers() & QtCore.Qt.ControlModifier:
                                # delete mask selected
                                self.parent.remove_cell(idx)
                            elif ev.modifiers() & QtCore.Qt.AltModifier:
                                self.parent.merge_cells(idx)
                            elif self.parent.masksOn:
                                self.parent.unselect_cell()
                                self.parent.select_cell(idx)
                        elif self.parent.masksOn:
                            self.parent.unselect_cell()

    def mouseDragEvent(self, ev):
        ev.ignore()
        return

    def hoverEvent(self, ev):
        #QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
        if self.parent.in_stroke:
            if self.parent.in_stroke:
                # continue stroke if not at start
                self.drawAt(ev.pos())
                if self.is_at_start(ev.pos()):
                    self.end_stroke()
                    
        else:
            ev.acceptClicks(QtCore.Qt.RightButton)
            #ev.acceptClicks(QtCore.Qt.LeftButton)

    def create_start(self, pos):
        self.scatter = pg.ScatterPlotItem([pos.x()], [pos.y()], pxMode=False,
                                        pen=pg.mkPen(color=(255,0,0), width=self.parent.brush_size),
                                        size=max(3*2, self.parent.brush_size*1.8*2), brush=None)
        self.parent.p0.addItem(self.scatter)

    def is_at_start(self, pos):
        thresh_out = max(6, self.parent.brush_size*3)
        thresh_in = max(3, self.parent.brush_size*1.8)
        # first check if you ever left the start
        if len(self.parent.current_stroke) > 3:
            stroke = np.array(self.parent.current_stroke)
            dist = (((stroke[1:,1:] - stroke[:1,1:][np.newaxis,:,:])**2).sum(axis=-1))**0.5
            dist = dist.flatten()
            #print(dist)
            has_left = (dist > thresh_out).nonzero()[0]
            if len(has_left) > 0:
                first_left = np.sort(has_left)[0]
                has_returned = (dist[max(4,first_left+1):] < thresh_in).sum()
                if has_returned > 0:
                    return True
                else:
                    return False
            else:
                return False

    def end_stroke(self):
        self.parent.p0.removeItem(self.scatter)
        if not self.parent.stroke_appended:
            self.parent.strokes.append(self.parent.current_stroke)
            self.parent.stroke_appended = True
            self.parent.current_stroke = np.array(self.parent.current_stroke)
            ioutline = self.parent.current_stroke[:,3]==1
            self.parent.current_point_set.extend(list(self.parent.current_stroke[ioutline]))
            self.parent.current_stroke = []
            if self.parent.autosave:
                self.parent.add_set()
        if len(self.parent.current_point_set) > 0 and self.parent.autosave:
            self.parent.add_set()
        self.parent.in_stroke = False

    def tabletEvent(self, ev):
        pass
        #print(ev.device())
        #print(ev.pointerType())
        #print(ev.pressure())

    def drawAt(self, pos, ev=None):
        mask = self.strokemask
        set = self.parent.current_point_set
        stroke = self.parent.current_stroke
        pos = [int(pos.y()), int(pos.x())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0,dk.shape[0]]
        sy = [0,dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]
        kcent = kc.copy()
        if tx[0]<=0:
            sx[0] = 0
            sx[1] = kc[0] + 1
            tx    = sx
            kcent[0] = 0
        if ty[0]<=0:
            sy[0] = 0
            sy[1] = kc[1] + 1
            ty    = sy
            kcent[1] = 0
        if tx[1] >= self.parent.Ly-1:
            sx[0] = dk.shape[0] - kc[0] - 1
            sx[1] = dk.shape[0]
            tx[0] = self.parent.Ly - kc[0] - 1
            tx[1] = self.parent.Ly
            kcent[0] = tx[1]-tx[0]-1
        if ty[1] >= self.parent.Lx-1:
            sy[0] = dk.shape[1] - kc[1] - 1
            sy[1] = dk.shape[1]
            ty[0] = self.parent.Lx - kc[1] - 1
            ty[1] = self.parent.Lx
            kcent[1] = ty[1]-ty[0]-1


        ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
        ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))
        self.image[ts] = mask[ss]

        for ky,y in enumerate(np.arange(ty[0], ty[1], 1, int)):
            for kx,x in enumerate(np.arange(tx[0], tx[1], 1, int)):
                iscent = np.logical_and(kx==kcent[0], ky==kcent[1])
                stroke.append([self.parent.currentZ, x, y, iscent])
        self.updateImage()

    def setDrawKernel(self, kernel_size=3):
        bs = kernel_size
        kernel = np.ones((bs,bs), np.uint8)
        self.drawKernel = kernel
        self.drawKernelCenter = [int(np.floor(kernel.shape[0]/2)),
                                 int(np.floor(kernel.shape[1]/2))]
        onmask = 255 * kernel[:,:,np.newaxis]
        offmask = np.zeros((bs,bs,1))
        opamask = 100 * kernel[:,:,np.newaxis]
        self.redmask = np.concatenate((onmask,offmask,offmask,onmask), axis=-1)
        self.strokemask = np.concatenate((onmask,offmask,onmask,opamask), axis=-1)
