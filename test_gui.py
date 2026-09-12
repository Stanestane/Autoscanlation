"""Queue/progress integration tests; no GPU, model download, or translation service."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import tempfile
import threading
import time
import unittest
import zipfile
from unittest.mock import patch

from PIL import Image
from PySide6 import QtCore, QtWidgets

from gui import MainWindow, TranslationWorker, job_output_directory
from progress import report_progress


class TestPipeline:
    def __init__(self, cache, source_lang, target_lang, progress_callback):
        self.source_lang, self.target_lang = source_lang, target_lang
        self.progress_callback = progress_callback

    def analyze(self, path):
        report_progress(self, 'cache', 'Reusing cached OCR')
        return Image.open(path).convert('RGB'), dict(source=Path(path).name, bubbles=[])


def extract_zip(path, outdir, verbosity):
    with zipfile.ZipFile(path) as archive:
        archive.extractall(outdir)


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.page = self.root / 'page.png'
        Image.new('RGB', (60, 40), 'white').save(self.page)
        self.archive = self.root / 'volume.cbz'
        with zipfile.ZipFile(self.archive, 'w') as archive:
            archive.write(self.page, 'one.png')
            archive.write(self.page, 'two.png')

    def tearDown(self):
        self.temporary.cleanup()

    def tick(self):
        # Let the Python worker acquire the GIL between Qt event dispatches.
        self.app.processEvents()
        time.sleep(.01)

    def worker(self, jobs, max_pages=None):
        return TranslationWorker(jobs, self.root/'results', 'es', 'en', None, max_pages)

    def test_archive_progress_counts_saved_pages_and_finishes_after_verification(self):
        worker = self.worker([(0, self.archive), (1, self.page)])
        events = []
        worker.progress.connect(events.append)
        with patch('gui.Pipeline', TestPipeline), patch('patoolib.extract_archive', extract_zip):
            worker.run()
        first = [event for event in events if event.get('row') == 0]
        saved = [event for event in first if event['stage'] == 'page_saved']
        self.assertEqual([(e['pages_saved'], e['pages_total']) for e in saved], [(1,2), (2,2)])
        stages = [e['stage'] for e in first]
        self.assertLess(stages.index('pack'), stages.index('verify'))
        self.assertLess(stages.index('verify'), stages.index('job_done'))
        self.assertEqual(events[-1]['completed'], 2)
        second = [e for e in events if e.get('row') == 1 and e['stage'] == 'page_saved']
        self.assertEqual((second[0]['pages_saved'], second[0]['pages_total']), (1,1))
        output = next(e['output'] for e in first if e['stage'] == 'job_done')
        with zipfile.ZipFile(output) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()), 2)

    def test_page_limit_uses_selected_total(self):
        worker = self.worker([(0, self.archive)], max_pages=1)
        events = []
        worker.progress.connect(events.append)
        with patch('gui.Pipeline', TestPipeline), patch('patoolib.extract_archive', extract_zip):
            worker.run()
        saved = next(e for e in events if e['stage'] == 'page_saved')
        self.assertEqual((saved['pages_saved'], saved['pages_total']), (1, 1))

    def test_failed_file_does_not_block_next_file(self):
        worker = self.worker([(0, self.root/'missing.png'), (1, self.page)])
        events = []
        worker.progress.connect(events.append)
        with patch('gui.Pipeline', TestPipeline):
            worker.run()
        self.assertEqual([e['stage'] for e in events if e['stage'] in {'job_failed', 'job_done'}],
                         ['job_failed', 'job_done'])
        self.assertEqual((events[-1]['failed'], events[-1]['completed']), (1, 1))

    def test_stop_finishes_current_file_and_leaves_rest_pending(self):
        worker = self.worker([(0, self.archive), (1, self.page)])
        events = []
        def observe(event):
            events.append(event)
            if event['stage'] == 'page_saved':
                worker.stop_requested.set()
        worker.progress.connect(observe)
        with patch('gui.Pipeline', TestPipeline), patch('patoolib.extract_archive', extract_zip):
            worker.run()
        self.assertEqual((events[-1]['completed'], events[-1]['pending']), (1, 1))
        self.assertEqual(len([e for e in events if e['stage'] == 'page_saved']), 2)
        self.assertTrue(any(e['stage'] == 'verify' for e in events))

    def test_duplicate_paths_and_names(self):
        other = self.root / 'other' / 'page.png'
        other.parent.mkdir()
        Image.new('RGB', (20,20), 'white').save(other)
        window = MainWindow()
        try:
            window.add_files([self.page, self.page, other])
            self.assertEqual(window.queue.rowCount(), 2)
            self.assertNotEqual(job_output_directory(self.root, self.page),
                                job_output_directory(self.root, other))
        finally:
            window.close()

    def test_thread_delivers_live_events_and_closes_safely(self):
        entered, release = threading.Event(), threading.Event()
        class SlowPipeline(TestPipeline):
            def analyze(self, path):
                report_progress(self, 'ocr', 'Reading bubble 1 of 3')
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('Test timed out')
                return super().analyze(path)
        window = MainWindow()
        window.output_edit.setText(str(self.root/'results'))
        window.add_files([self.page])
        window.show()
        try:
            with patch('gui.Pipeline', SlowPipeline):
                window.start()
                for _ in range(200):
                    self.tick()
                    if entered.is_set() and 'Reading bubble 1 of 3' in window.log.toPlainText():
                        break
                self.assertTrue(entered.is_set())
                self.assertIn('Reading bubble 1 of 3', window.log.toPlainText())
                self.assertEqual(window.progress_bar.value(), 0)
                self.assertFalse(window.start_button.isEnabled())
                window.close()
                self.assertTrue(window.isVisible(), 'Closing while processing must be blocked')
                release.set()
                for _ in range(300):
                    self.tick()
                    if window.thread is None:
                        break
                self.assertIsNone(window.thread)
                self.assertEqual(window.queue.item(0, 1).text(), 'Completed')
                self.assertEqual(window.progress_bar.value(), 1)
                self.assertEqual(window.queue_progress.value(), 1)
                self.assertTrue(window.start_button.isEnabled())
        finally:
            release.set()
            for _ in range(300):
                if window.thread is None:
                    break
                self.tick()
            window.close()


if __name__ == '__main__':
    unittest.main()
