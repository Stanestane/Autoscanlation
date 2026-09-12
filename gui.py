"""Cross-platform desktop queue with page and processing-stage progress."""
import hashlib
from pathlib import Path
import sys
import threading

from PySide6 import QtCore, QtGui, QtWidgets

from comic_pipeline import Pipeline, ROOT
from translate_comic import IMAGE_EXTENSIONS, ARCHIVE_EXTENSIONS, process_archive, process_page


def job_output_directory(root, source):
    """Keep files with identical names in different source folders separate."""
    source = Path(source).resolve()
    digest = hashlib.sha256(str(source).encode()).hexdigest()[:10]
    return Path(root) / f'{source.name}-{digest}'


class TranslationWorker(QtCore.QObject):
    progress = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, jobs, output, source_lang, target_lang, font_size, max_pages):
        super().__init__()
        self.jobs = list(jobs)
        self.output = Path(output)
        self.source_lang, self.target_lang = source_lang, target_lang
        self.font_size, self.max_pages = font_size, max_pages
        self.stop_requested = threading.Event()
        self.current_row = -1
        self.pages_saved = self.pages_total = self.review_regions = 0

    def on_progress(self, event):
        if event['stage'] == 'pages':
            self.pages_total = event['total']
        elif event['stage'] == 'page_saved':
            self.pages_saved += 1
            self.review_regions += sum(value for status, value in event['summary'].items()
                                       if status not in {'translated', 'not_dialogue'})
        self.progress.emit(dict(event, row=self.current_row, pages_saved=self.pages_saved,
                                pages_total=self.pages_total))

    @QtCore.Slot()
    def run(self):
        pipeline = None
        succeeded = failed = 0
        try:
            for index, (row, source) in enumerate(self.jobs):
                if self.stop_requested.is_set():
                    break
                source = Path(source)
                self.current_row = row
                self.pages_saved = self.review_regions = 0
                self.pages_total = 1 if source.suffix.lower() in IMAGE_EXTENSIONS else 0
                self.on_progress(dict(stage='job_start', message=f'File {index+1} of {len(self.jobs)}: {source.name}'))
                try:
                    # Construct cache/model state in the worker, and reuse it for the queue.
                    if pipeline is None:
                        pipeline = Pipeline(ROOT / 'output/.cache/gui', self.source_lang,
                                            self.target_lang, progress_callback=self.on_progress)
                    destination = job_output_directory(self.output, source)
                    if source.suffix.lower() in ARCHIVE_EXTENSIONS:
                        result = process_archive(source, destination, pipeline, self.font_size, self.max_pages)
                    elif source.suffix.lower() in IMAGE_EXTENSIONS:
                        result = destination / f'translated_{source.name}.png'
                        process_page(source, result, pipeline, self.font_size)
                    else:
                        raise ValueError('Unsupported input file')
                    succeeded += 1
                    message = f'Created {result}'
                    if self.review_regions:
                        message += f' — {self.review_regions} regions need review; see the JSON reports.'
                    self.on_progress(dict(stage='job_done', message=message, output=str(result),
                                          review_regions=self.review_regions))
                except Exception as exc:
                    failed += 1
                    self.on_progress(dict(stage='job_failed', message=f'{type(exc).__name__}: {exc}'))
            pending = len(self.jobs) - succeeded - failed
            self.progress.emit(dict(stage='queue_done', message=f'{succeeded} completed, {failed} failed, {pending} pending.',
                                    completed=succeeded, failed=failed, pending=pending))
        finally:
            self.finished.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Comic Translator')
        self.resize(1000, 760)
        self.thread = self.worker = None
        self.jobs_processed = 0
        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)
        title = QtWidgets.QLabel('Translation queue')
        font = title.font()
        font.setPointSize(18)
        font.setBold(True)
        title.setFont(font)
        root.addWidget(title)

        self.add_button = QtWidgets.QPushButton('Add files…')
        self.add_button.clicked.connect(self.choose_input)
        self.remove_button = QtWidgets.QPushButton('Remove selected')
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button = QtWidgets.QPushButton('Clear queue')
        self.clear_button.clicked.connect(self.clear_queue)
        root.addWidget(self._row(self.add_button, self.remove_button, self.clear_button))

        self.queue = QtWidgets.QTableWidget(0, 3)
        self.queue.setHorizontalHeaderLabels(['File', 'Status', 'Saved pages'])
        self.queue.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.queue.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.queue.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.queue.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.queue.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.queue.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue.setAlternatingRowColors(True)
        root.addWidget(self.queue, 2)

        self.settings = QtWidgets.QGroupBox('Settings for queued files')
        form = QtWidgets.QFormLayout(self.settings)
        self.output_edit = QtWidgets.QLineEdit(str(ROOT / 'output/gui_runs'))
        choose = QtWidgets.QPushButton('Choose…')
        choose.clicked.connect(self.choose_output)
        form.addRow('Output folder', self._row(self.output_edit, choose))
        self.source_lang = QtWidgets.QLineEdit('es')
        self.target_lang = QtWidgets.QLineEdit('en')
        form.addRow('Languages', self._row(QtWidgets.QLabel('Source'), self.source_lang,
                                         QtWidgets.QLabel('Target'), self.target_lang))
        self.font_size = QtWidgets.QSpinBox()
        self.font_size.setRange(0, 120)
        self.font_size.setSpecialValueText('Auto')
        self.max_pages = QtWidgets.QSpinBox()
        self.max_pages.setRange(0, 9999)
        self.max_pages.setSpecialValueText('All pages')
        form.addRow('Options', self._row(QtWidgets.QLabel('Max font'), self.font_size,
                                       QtWidgets.QLabel('Pages per archive'), self.max_pages))
        root.addWidget(self.settings)

        self.start_button = QtWidgets.QPushButton('Start / retry queue')
        self.start_button.setMinimumHeight(38)
        self.start_button.clicked.connect(self.start)
        self.stop_button = QtWidgets.QPushButton('Stop after current file')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_after_file)
        self.open_button = QtWidgets.QPushButton('Open output folder')
        self.open_button.clicked.connect(self.open_output)
        root.addWidget(self._row(self.start_button, self.stop_button, self.open_button))
        self.queue_progress = QtWidgets.QProgressBar()
        self.queue_progress.setRange(0, 1)
        self.queue_progress.setValue(0)
        self.queue_progress.setFormat('No files running')
        root.addWidget(self.queue_progress)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('Pages saved: 0')
        root.addWidget(self.progress_bar)
        self.stage_label = QtWidgets.QLabel('Add images or CBZ/CBR archives to get started.')
        self.stage_label.setWordWrap(True)
        root.addWidget(self.stage_label)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        root.addWidget(self.log, 1)

    @staticmethod
    def _row(*widgets):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        return container

    def add_files(self, paths):
        if self.thread is not None:
            return
        existing = {self.queue.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
                    for row in range(self.queue.rowCount())}
        for value in paths:
            path = Path(value).resolve()
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS | ARCHIVE_EXTENSIONS:
                self.log.appendPlainText(f'Skipped unsupported or missing file: {path}')
                continue
            if path in existing:
                continue
            existing.add(path)
            row = self.queue.rowCount()
            self.queue.insertRow(row)
            item = QtWidgets.QTableWidgetItem(path.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            item.setToolTip(str(path))
            self.queue.setItem(row, 0, item)
            self.queue.setItem(row, 1, QtWidgets.QTableWidgetItem('Pending'))
            self.queue.setItem(row, 2, QtWidgets.QTableWidgetItem('—'))

    def choose_input(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, 'Add comic archives or pages', str(ROOT / 'input'),
            'Comics and images (*.cbz *.cbr *.jpg *.jpeg *.png *.webp *.bmp *.gif *.tif *.tiff)')
        self.add_files(paths)

    def remove_selected(self):
        if self.thread is None:
            for row in sorted({index.row() for index in self.queue.selectedIndexes()}, reverse=True):
                self.queue.removeRow(row)

    def clear_queue(self):
        if self.thread is None:
            self.queue.setRowCount(0)

    def choose_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose output folder', self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def open_output(self):
        path = Path(self.output_edit.text()).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def set_running(self, running):
        for widget in (self.add_button, self.remove_button, self.clear_button, self.settings, self.start_button):
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def start(self):
        if self.thread is not None:
            return
        jobs = [(row, self.queue.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole))
                for row in range(self.queue.rowCount()) if self.queue.item(row, 1).text() in {'Pending', 'Failed'}]
        if not jobs:
            self.stage_label.setText('Add files to the queue. Completed files can be removed and added again to rerun.')
            return
        if not self.output_edit.text().strip():
            self.stage_label.setText('Choose an output folder before starting.')
            return
        for row, _ in jobs:
            self.queue.item(row, 1).setText('Pending')
            self.queue.item(row, 2).setText('—')
        self.jobs_processed = 0
        self.queue_progress.setRange(0, len(jobs))
        self.queue_progress.setValue(0)
        self.queue_progress.setFormat('Files processed: %v / %m')
        self.set_running(True)
        self.thread = QtCore.QThread(self)
        self.worker = TranslationWorker(jobs, Path(self.output_edit.text()).expanduser(),
                                        self.source_lang.text().strip() or 'auto',
                                        self.target_lang.text().strip() or 'en',
                                        self.font_size.value() or None, self.max_pages.value() or None)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @QtCore.Slot(object)
    def on_progress(self, event):
        stage, message = event['stage'], event['message']
        self.stage_label.setText(message)
        self.log.appendPlainText(message)
        if stage == 'queue_done':
            return
        row = event['row']
        status = self.queue.item(row, 1)
        saved, total = event['pages_saved'], event['pages_total']
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(saved)
        self.progress_bar.setFormat('Pages saved: %v / %m' if total else 'Reading archive…')
        self.queue.item(row, 2).setText(f'{saved} / {total}' if total else '—')
        if stage == 'job_start':
            status.setText('Running')
            self.queue.selectRow(row)
        elif stage in {'pack', 'verify'}:
            status.setText('Creating archive' if stage == 'pack' else 'Verifying archive')
        elif stage in {'job_done', 'job_failed'}:
            status.setText('Failed' if stage == 'job_failed' else (
                'Completed — review needed' if event['review_regions'] else 'Completed'))
            status.setToolTip(message)
            if stage == 'job_failed' and not total:
                self.progress_bar.setRange(0, 1)
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat('File failed before page processing')
            self.jobs_processed += 1
            self.queue_progress.setValue(self.jobs_processed)

    def stop_after_file(self):
        if self.worker is not None:
            self.worker.stop_requested.set()
            self.stop_button.setEnabled(False)
            self.log.appendPlainText('Queue will stop after the current file has finished exporting.')

    @QtCore.Slot()
    def thread_finished(self):
        self.thread = self.worker = None
        self.set_running(False)

    def closeEvent(self, event):
        if self.thread is not None:
            self.stage_label.setText('A file is still processing. Use “Stop after current file” and wait before closing.')
            event.ignore()
        else:
            event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('Comic Translator')
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
