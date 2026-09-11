"""Cross-platform PySide6 desktop interface for Comic Translator."""
from pathlib import Path
import sys

from PySide6 import QtCore, QtWidgets

from comic_pipeline import Pipeline, ROOT
from translate_comic import IMAGE_EXTENSIONS, ARCHIVE_EXTENSIONS, process_archive, process_page


class TranslationWorker(QtCore.QObject):
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, source, output, source_lang, target_lang, font_size, max_pages):
        super().__init__()
        self.source = Path(source)
        self.output = Path(output)
        self.pipeline = Pipeline(ROOT / 'output/.cache/gui', source_lang, target_lang)
        self.font_size, self.max_pages = font_size, max_pages

    @QtCore.Slot()
    def run(self):
        try:
            self.progress.emit('Loading models on the first run can take several minutes...')
            if self.source.suffix.lower() in ARCHIVE_EXTENSIONS:
                result = process_archive(self.source, self.output, self.pipeline,
                                         self.font_size, self.max_pages)
            elif self.source.suffix.lower() in IMAGE_EXTENSIONS:
                result = self.output / f'translated_{self.source.stem}.png'
                process_page(self.source, result, self.pipeline, self.font_size)
            else:
                raise ValueError('Choose an image, CBZ, or CBR file')
            self.finished.emit(str(result))
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Comic Translator')
        self.resize(900, 600)
        self.thread = self.worker = None
        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central); form = QtWidgets.QFormLayout()
        self.input_edit = QtWidgets.QLineEdit()
        choose = QtWidgets.QPushButton('Choose...'); choose.clicked.connect(self.choose_input)
        form.addRow('Input archive or page', self._row(self.input_edit, choose))
        self.output_edit = QtWidgets.QLineEdit(str(ROOT / 'output/gui_runs'))
        choose = QtWidgets.QPushButton('Choose...'); choose.clicked.connect(self.choose_output)
        form.addRow('Output folder', self._row(self.output_edit, choose))
        self.source_lang = QtWidgets.QLineEdit('es'); self.target_lang = QtWidgets.QLineEdit('en')
        langs = self._row(QtWidgets.QLabel('Source'), self.source_lang, QtWidgets.QLabel('Target'), self.target_lang)
        form.addRow('Languages', langs)
        self.font_size = QtWidgets.QSpinBox(); self.font_size.setRange(0, 120); self.font_size.setValue(0); self.font_size.setSpecialValueText('Auto')
        self.max_pages = QtWidgets.QSpinBox(); self.max_pages.setRange(0, 9999); self.max_pages.setValue(0); self.max_pages.setSpecialValueText('All pages')
        form.addRow('Options', self._row(QtWidgets.QLabel('Max font'), self.font_size, QtWidgets.QLabel('Sample pages'), self.max_pages))
        root.addLayout(form)
        self.start_button = QtWidgets.QPushButton('Translate'); self.start_button.setMinimumHeight(42); self.start_button.clicked.connect(self.start)
        root.addWidget(self.start_button)
        self.progress_bar = QtWidgets.QProgressBar(); self.progress_bar.setRange(0, 0); self.progress_bar.hide(); root.addWidget(self.progress_bar)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); root.addWidget(self.log, 1)

    @staticmethod
    def _row(*widgets):
        layout = QtWidgets.QHBoxLayout(); layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets: layout.addWidget(widget)
        container = QtWidgets.QWidget(); container.setLayout(layout); return container

    def choose_input(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose comic or page', str(ROOT / 'input'), 'Comics and images (*.cbz *.cbr *.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)')
        if path: self.input_edit.setText(path)

    def choose_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose output folder', self.output_edit.text())
        if path: self.output_edit.setText(path)

    def start(self):
        source = Path(self.input_edit.text().strip())
        if not source.is_file():
            QtWidgets.QMessageBox.warning(self, 'Input required', 'Choose an existing comic archive or image.'); return
        self.start_button.setEnabled(False); self.progress_bar.show(); self.log.appendPlainText(f'Starting: {source.name}')
        self.thread = QtCore.QThread(self)
        self.worker = TranslationWorker(source, Path(self.output_edit.text()), self.source_lang.text().strip() or 'auto', self.target_lang.text().strip() or 'en', self.font_size.value() or None, self.max_pages.value() or None)
        self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.log.appendPlainText); self.worker.finished.connect(self.completed); self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.thread.quit); self.worker.failed.connect(self.thread.quit); self.thread.finished.connect(self.thread_finished); self.thread.start()

    def completed(self, output):
        self.log.appendPlainText(f'Finished: {output}'); QtWidgets.QMessageBox.information(self, 'Translation complete', f'Created:\n{output}')

    def failed(self, message):
        self.log.appendPlainText(f'ERROR: {message}'); QtWidgets.QMessageBox.critical(self, 'Translation failed', message)

    def thread_finished(self):
        self.progress_bar.hide(); self.start_button.setEnabled(True); self.worker.deleteLater(); self.thread.deleteLater(); self.worker = self.thread = None


def main():
    app = QtWidgets.QApplication(sys.argv); app.setApplicationName('Comic Translator')
    window = MainWindow(); window.show(); return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
