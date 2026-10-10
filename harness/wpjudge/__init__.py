"""wpjudge: WordPress plugin/theme -> per-function judge -> JSON report."""
import tempfile


def scan(path, on_result=None, **cfg):
    """Library one-shot: archive or dir -> report dict.

    on_result(judgment, done, total) is called once with (None, 0, total) after extraction,
    then once per finished judgment (judge_units calls it from the calling thread).
    """
    from . import aggregate, extract, judge
    from .config import load

    cfg = load(**cfg)
    with tempfile.TemporaryDirectory(prefix="wpjudge-") as tmp:
        root, target_meta = extract.load_target(path, tmp, cfg)
        units, skips, files_scanned = extract.collect_units(root, cfg)
        total, done, cb = len(units), [0], None
        if on_result:
            on_result(None, 0, total)

            def cb(j):
                done[0] += 1
                on_result(j, done[0], total)
        # llama-server ignores the model name; the file name keeps logs readable
        judgments = judge.judge_units(units, cfg.base_url, cfg.model_file, cfg.workers,
                                      on_result=cb, max_tokens=cfg.max_tokens)
        model_meta = {"file": cfg.model_file, "base_url": cfg.base_url}
        return aggregate.build_report(target_meta, model_meta, judgments, skips, cfg,
                                      files_scanned=files_scanned)


__all__ = ["scan"]
