# SPDX-FileCopyrightText: 2026 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for MinerU OCR engine integration."""

from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from ocrmypdf import OcrClass


def _make_options(**overrides):
    defaults = dict(
        ocr_engine='mineru',
        pdf_renderer='auto',
        languages=['eng'],
        mineru_api_token='token',
        mineru_base_url='https://mineru.example/api/v4',
        mineru_model_version='vlm',
        mineru_language='ch',
        mineru_enable_formula=True,
        mineru_enable_table=True,
        mineru_poll_interval=0.1,
        mineru_timeout=5.0,
        mineru_include_discarded=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_layout_zip(layout_payload: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('layout.json', json.dumps(layout_payload))
    return buffer.getvalue()


def test_parse_layout_result():
    from ocrmypdf.builtin_plugins.mineru_ocr import _parse_layout_result

    layout = {
        'pdf_info': [
            {
                'page_idx': 0,
                'page_size': [200, 100],
                'para_blocks': [
                    {
                        'bbox': [10, 20, 190, 50],
                        'lines': [
                            {
                                'bbox': [10, 20, 190, 50],
                                'spans': [
                                    {
                                        'bbox': [10, 20, 80, 50],
                                        'type': 'text',
                                        'content': 'Hello',
                                        'score': 0.95,
                                    },
                                    {
                                        'bbox': [90, 20, 190, 50],
                                        'type': 'text',
                                        'content': 'World',
                                        'score': 0.9,
                                    },
                                ],
                            }
                        ],
                    }
                ],
                'discarded_blocks': [],
            }
        ]
    }

    pages = _parse_layout_result(layout, include_discarded=False)
    assert len(pages) == 1

    page, text = pages[0]
    assert page.ocr_class == OcrClass.PAGE
    assert page.bbox is not None
    assert page.bbox.right == 200
    assert page.bbox.bottom == 100
    assert len(page.words) == 2
    assert text == 'Hello World'


def test_generate_ocr_uses_layout(monkeypatch, tmp_path):
    from PIL import Image

    from ocrmypdf.builtin_plugins.mineru_ocr import MinerUOcrEngine

    image_path = tmp_path / 'page.png'
    image = Image.new('RGB', (300, 200), 'white')
    image.save(image_path, dpi=(300, 300))

    layout = {
        'pdf_info': [
            {
                'page_idx': 0,
                'page_size': [300, 200],
                'para_blocks': [
                    {
                        'bbox': [40, 60, 260, 100],
                        'lines': [
                            {
                                'bbox': [40, 60, 260, 100],
                                'spans': [
                                    {
                                        'bbox': [40, 60, 150, 100],
                                        'type': 'text',
                                        'content': 'MinerU',
                                        'score': 1.0,
                                    },
                                    {
                                        'bbox': [160, 60, 260, 100],
                                        'type': 'text',
                                        'content': 'OCR',
                                        'score': 1.0,
                                    },
                                ],
                            }
                        ],
                    }
                ],
                'discarded_blocks': [],
            }
        ]
    }
    zip_payload = _make_layout_zip(layout)

    monkeypatch.setattr(
        'ocrmypdf.builtin_plugins.mineru_ocr._run_mineru_api',
        lambda settings, input_file: zip_payload,
    )

    page, text = MinerUOcrEngine.generate_ocr(
        input_file=image_path,
        options=_make_options(),
        page_number=0,
    )

    assert page.ocr_class == OcrClass.PAGE
    assert [word.text for word in page.words] == ['MinerU', 'OCR']
    assert text == 'MinerU OCR'


def test_check_options_requires_token():
    from ocrmypdf.builtin_plugins.mineru_ocr import check_options
    from ocrmypdf.exceptions import BadArgsError

    with pytest.raises(BadArgsError):
        check_options(_make_options(mineru_api_token=''))


def test_check_options_rejects_sandwich_renderer():
    from ocrmypdf.builtin_plugins.mineru_ocr import check_options
    from ocrmypdf.exceptions import BadArgsError

    with pytest.raises(BadArgsError):
        check_options(_make_options(pdf_renderer='sandwich'))


def test_generate_ocr_returns_empty_page_when_no_result(monkeypatch, tmp_path):
    from PIL import Image

    from ocrmypdf.builtin_plugins.mineru_ocr import MinerUOcrEngine

    image_path = tmp_path / 'page.png'
    image = Image.new('RGB', (120, 80), 'white')
    image.save(image_path, dpi=(300, 300))

    empty_zip = io.BytesIO()
    with zipfile.ZipFile(empty_zip, 'w') as archive:
        archive.writestr('full.md', 'no layout')

    monkeypatch.setattr(
        'ocrmypdf.builtin_plugins.mineru_ocr._run_mineru_api',
        lambda settings, input_file: empty_zip.getvalue(),
    )

    page, text = MinerUOcrEngine.generate_ocr(
        input_file=image_path,
        options=_make_options(),
        page_number=0,
    )

    assert page.ocr_class == OcrClass.PAGE
    assert page.bbox is not None
    assert page.bbox.right == 120
    assert page.bbox.bottom == 80
    assert text == ''
