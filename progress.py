"""Optional structured progress events shared by the CLI pipeline and GUI."""


def report_progress(pipeline, stage, message, **details):
    callback = getattr(pipeline, 'progress_callback', None)
    if callback is not None:
        callback(dict(stage=stage, message=message, **details))
