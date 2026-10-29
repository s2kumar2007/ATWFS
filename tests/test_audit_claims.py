import json
from scripts.check_metric_claims import check_document


def test_metric_claim_changes_and_missing_report_fail(tmp_path):
    (tmp_path/'report.json').write_text(json.dumps({'mIoU':.9}))
    doc=tmp_path/'claim.md'
    doc.write_text('| mIoU | 0.900 | <!-- metric: report.json#/mIoU = 0.900 -->')
    assert check_document(doc,tmp_path)==[]
    (tmp_path/'report.json').write_text(json.dumps({'mIoU':.8}))
    assert check_document(doc,tmp_path)
    (tmp_path/'report.json').unlink()
    assert check_document(doc,tmp_path)


def test_quality_guard_preserves_threshold_and_fails_empty():
    from scripts.verify_quality_guard import quality_guard_passed
    assert quality_guard_passed([.97,.99])
    assert not quality_guard_passed([.5,.6])
    assert not quality_guard_passed([])
