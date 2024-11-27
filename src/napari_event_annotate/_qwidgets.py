from PyQt5.QtWidgets import QComboBox, QVBoxLayout, QWidget, QApplication, QMainWindow
from PyQt5.QtCore import QSettings

class HistoryComboBox(QComboBox):
    def __init__(self, name, parent=None):
        super().__init__(parent)

        self.setEditable(True)
        self.setInsertPolicy(1)
        # Initialize QSettings
        self.settings = QSettings("EPFL", name)

        # Load history from QSettings
        self.load_history()

        self.lineEdit().editingFinished.connect(self.update_history)
        self.activated.connect(self.update_history_highlight)
        
    def load_history(self):
        self.history = self.settings.value("comboBoxHistory", [])
        if self.history:
            for item in self.history:
                self.addItem(item)

    def update_history_highlight(self, i):
        text = self.itemText(i)
        self.removeItem(i)
        self.insertItem(0, text)
        self.setCurrentIndex(0)

    def update_history(self):
        if self.count() > 5:
            self.removeItem(self.count()-1)

    def save_history(self):
        history = [self.itemText(i) for i in range(self.count())]
        self.settings.setValue("comboBoxHistory", history)
    


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.historyComboBox = HistoryComboBox("test")

        layout = QVBoxLayout()
        layout.addWidget(self.historyComboBox)
        
        container = QWidget()
        container.setLayout(layout)
        
        self.setCentralWidget(container)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
