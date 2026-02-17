# SPDX-FileCopyrightText: 2026 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Built-in plugin to implement OCR using MinerU API."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
import uuid
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib import error as urlerror
from urllib import request as urlrequest

from PIL import Image

from ocrmypdf import hookimpl
from ocrmypdf.cli import numeric
from ocrmypdf.exceptions import BadArgsError
from ocrmypdf.hocrtransform import BoundingBox, OcrClass, OcrElement
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

if TYPE_CHECKING:
    from ocrmypdf._options import OcrOptions

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://mineru.net/api/v4"
_DEFAULT_MODEL = "vlm"
_DEFAULT_LANGUAGE = "ch"


@dataclass(frozen=True)
class MinerUSettings:
    """Runtime settings for MinerU API calls."""

    api_token: str
    base_url: str
    model_version: str
    language: str
    enable_formula: bool
    enable_table: bool
    poll_interval: float
    timeout: float
    include_discarded: bool


def _has_cjk(text: str) -> bool:
    return any('\u4e00' <= char <= '\u9fff' for char in text)


def _join_text(tokens: list[str]) -> str:
    if not tokens:
        return ""
    if any(_has_cjk(token) for token in tokens):
        return ''.join(tokens)
    return ' '.join(tokens)


def _coerce_bbox(raw: Any) -> BoundingBox | None:
    if not isinstance(raw, list | tuple) or len(raw) < 4:
        return None
    try:
        left = float(raw[0])
        top = float(raw[1])
        right = float(raw[2])
        bottom = float(raw[3])
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return BoundingBox(left=left, top=top, right=right, bottom=bottom)


def _normalize_confidence(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if score < 0:
        return 0.0
    if score <= 1:
        return score
    if score <= 100:
        return score / 100.0
    return 1.0


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        for key in ('paragraph_content', 'title_content', 'content', 'text', 'latex'):
            if key in content and isinstance(content[key], str):
                text = content[key].strip()
                if text:
                    return text
        if isinstance(content.get('list_items'), list):
            lines = [str(item).strip() for item in content['list_items']]
            lines = [line for line in lines if line]
            if lines:
                return '\n'.join(lines)
    return ""


def _parse_layout_result(
    layout_data: dict[str, Any], *, include_discarded: bool
) -> list[tuple[OcrElement, str]]:
    pages: list[tuple[OcrElement, str]] = []
    pdf_info = layout_data.get('pdf_info')
    if not isinstance(pdf_info, list):
        return pages

    for index, page_info in enumerate(pdf_info):
        if not isinstance(page_info, dict):
            continue

        page_size = page_info.get('page_size')
        page_width = page_height = 0.0
        if isinstance(page_size, list | tuple) and len(page_size) >= 2:
            try:
                page_width = float(page_size[0])
                page_height = float(page_size[1])
            except (TypeError, ValueError):
                page_width = page_height = 0.0

        blocks: list[Any] = list(page_info.get('para_blocks') or [])
        if include_discarded:
            blocks.extend(page_info.get('discarded_blocks') or [])

        max_right = max(page_width, 1.0)
        max_bottom = max(page_height, 1.0)
        page_lines: list[str] = []
        page_children: list[OcrElement] = []

        for block in blocks:
            if not isinstance(block, dict):
                continue

            para_bbox = _coerce_bbox(block.get('bbox'))
            if para_bbox is not None:
                max_right = max(max_right, para_bbox.right)
                max_bottom = max(max_bottom, para_bbox.bottom)
            para = OcrElement(
                ocr_class=OcrClass.PARAGRAPH,
                bbox=para_bbox,
            )

            lines = block.get('lines')
            if isinstance(lines, list):
                for line in lines:
                    if not isinstance(line, dict):
                        continue
                    line_bbox = _coerce_bbox(line.get('bbox')) or para_bbox
                    if line_bbox is not None:
                        max_right = max(max_right, line_bbox.right)
                        max_bottom = max(max_bottom, line_bbox.bottom)
                    line_elem = OcrElement(
                        ocr_class=OcrClass.LINE,
                        bbox=line_bbox,
                    )
                    spans = line.get('spans')
                    span_texts: list[str] = []
                    if isinstance(spans, list):
                        for span in spans:
                            if not isinstance(span, dict):
                                continue
                            text = str(span.get('content', '')).strip()
                            if not text:
                                continue
                            span_bbox = _coerce_bbox(span.get('bbox')) or line_bbox
                            if span_bbox is not None:
                                max_right = max(max_right, span_bbox.right)
                                max_bottom = max(max_bottom, span_bbox.bottom)
                            line_elem.children.append(
                                OcrElement(
                                    ocr_class=OcrClass.WORD,
                                    bbox=span_bbox,
                                    text=text,
                                    confidence=_normalize_confidence(span.get('score')),
                                )
                            )
                            span_texts.append(text)

                    if line_elem.children:
                        para.children.append(line_elem)
                        page_lines.append(_join_text(span_texts))

            if not para.children:
                fallback = _extract_text_from_content(block.get('content'))
                if fallback:
                    line_bbox = para_bbox
                    line_elem = OcrElement(ocr_class=OcrClass.LINE, bbox=line_bbox)
                    line_elem.children.append(
                        OcrElement(
                            ocr_class=OcrClass.WORD,
                            bbox=line_bbox,
                            text=fallback,
                        )
                    )
                    para.children.append(line_elem)
                    page_lines.append(fallback)

            if para.children:
                page_children.append(para)

        page_number = page_info.get('page_idx')
        if not isinstance(page_number, int):
            page_number = index

        page = OcrElement(
            ocr_class=OcrClass.PAGE,
            bbox=BoundingBox(left=0, top=0, right=max_right, bottom=max_bottom),
            dpi=72.0,
            page_number=page_number,
            children=page_children,
        )
        pages.append((page, '\n'.join(line for line in page_lines if line)))

    return pages


def _flatten_content_list_v2(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list) or not data:
        return []
    if not isinstance(data[0], list):
        return [item for item in data if isinstance(item, dict)]

    flattened: list[dict[str, Any]] = []
    for page_index, page_blocks in enumerate(data):
        if not isinstance(page_blocks, list):
            continue
        for block in page_blocks:
            if not isinstance(block, dict):
                continue
            text = _extract_text_from_content(block.get('content'))
            if not text:
                continue
            flattened.append(
                {
                    'page_idx': page_index,
                    'bbox': block.get('bbox'),
                    'text': text,
                }
            )
    return flattened


def _parse_content_list_result(
    content_data: Any,
    *,
    fallback_width: float,
    fallback_height: float,
    fallback_dpi: float,
) -> list[tuple[OcrElement, str]]:
    entries = _flatten_content_list_v2(content_data)
    if not entries:
        return []

    by_page: dict[int, list[tuple[BoundingBox | None, str]]] = defaultdict(list)
    for item in entries:
        page_idx = item.get('page_idx', 0)
        if not isinstance(page_idx, int):
            page_idx = 0
        text = str(item.get('text', '')).strip()
        if not text:
            continue
        by_page[page_idx].append((_coerce_bbox(item.get('bbox')), text))

    pages: list[tuple[OcrElement, str]] = []
    for page_idx in sorted(by_page):
        lines = by_page[page_idx]
        max_right = max(fallback_width, 1.0)
        max_bottom = max(fallback_height, 1.0)
        line_texts: list[str] = []
        paragraphs: list[OcrElement] = []

        for bbox, text in lines:
            if bbox is not None:
                max_right = max(max_right, bbox.right)
                max_bottom = max(max_bottom, bbox.bottom)

            line = OcrElement(ocr_class=OcrClass.LINE, bbox=bbox)
            line.children.append(
                OcrElement(ocr_class=OcrClass.WORD, bbox=bbox, text=text)
            )
            paragraph = OcrElement(
                ocr_class=OcrClass.PARAGRAPH,
                bbox=bbox,
                children=[line],
            )
            paragraphs.append(paragraph)
            line_texts.append(text)

        page = OcrElement(
            ocr_class=OcrClass.PAGE,
            bbox=BoundingBox(left=0, top=0, right=max_right, bottom=max_bottom),
            dpi=fallback_dpi,
            page_number=page_idx,
            children=paragraphs,
        )
        pages.append((page, '\n'.join(line_texts)))

    return pages


def _read_json_response(response_bytes: bytes) -> dict[str, Any]:
    payload = json.loads(response_bytes.decode('utf-8'))
    if not isinstance(payload, dict):
        raise RuntimeError("MinerU API returned a non-object JSON payload")
    return payload


def _http_json(
    *,
    url: str,
    method: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    encoded = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urlrequest.Request(url=url, data=encoded, method=method)
    for key, value in headers.items():
        req.add_header(key, value)
    if payload is not None:
        req.add_header('Content-Type', 'application/json')
    with urlrequest.urlopen(req, timeout=timeout) as response:
        body = response.read()
    return _read_json_response(body)


def _http_put_file(url: str, input_file: Path, timeout: float) -> None:
    req = urlrequest.Request(url=url, data=input_file.read_bytes(), method='PUT')
    with urlrequest.urlopen(req, timeout=timeout):
        return


def _http_get_bytes(url: str, timeout: float) -> bytes:
    with urlrequest.urlopen(url, timeout=timeout) as response:
        return response.read()


def _api_field(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _run_mineru_api(settings: MinerUSettings, input_file: Path) -> bytes:
    log.debug("Submitting file to MinerU API: %s", input_file.name)
    headers = {
        'Authorization': f'Bearer {settings.api_token}',
        'Accept': 'application/json',
    }
    base_url = settings.base_url.rstrip('/')
    request_timeout = max(1.0, min(settings.timeout, 120.0))

    submit_payload = {
        'files': [
            {
                'name': input_file.name,
                'data_id': uuid.uuid4().hex,
                'is_ocr': True,
            }
        ],
        'model_version': settings.model_version,
        'enable_formula': settings.enable_formula,
        'enable_table': settings.enable_table,
        'language': settings.language,
    }

    try:
        submit_resp = _http_json(
            url=f'{base_url}/file-urls/batch',
            method='POST',
            headers=headers,
            payload=submit_payload,
            timeout=request_timeout,
        )
    except (urlerror.HTTPError, urlerror.URLError, TimeoutError) as exc:
        raise RuntimeError(f"MinerU submit failed: {exc}") from exc

    if submit_resp.get('code') != 0:
        raise RuntimeError(f"MinerU submit failed: {submit_resp.get('msg', 'unknown')}")

    data = submit_resp.get('data')
    if not isinstance(data, dict):
        raise RuntimeError("MinerU submit failed: missing response data")

    batch_id = _api_field(data, 'batch_id')
    if not isinstance(batch_id, str) or not batch_id:
        raise RuntimeError("MinerU submit failed: missing batch_id")

    file_urls = _api_field(data, 'file_urls', 'files')
    if not isinstance(file_urls, list) or not file_urls:
        raise RuntimeError("MinerU submit failed: missing upload URL")
    upload_url = file_urls[0]
    if not isinstance(upload_url, str) or not upload_url:
        raise RuntimeError("MinerU submit failed: invalid upload URL")

    try:
        _http_put_file(upload_url, input_file, timeout=request_timeout)
    except (urlerror.HTTPError, urlerror.URLError, TimeoutError) as exc:
        raise RuntimeError(f"MinerU upload failed: {exc}") from exc

    poll_url = f'{base_url}/extract-results/batch/{batch_id}'
    deadline = time.monotonic() + settings.timeout
    zip_url: str | None = None

    while time.monotonic() < deadline:
        try:
            query_resp = _http_json(
                url=poll_url,
                method='GET',
                headers=headers,
                payload=None,
                timeout=request_timeout,
            )
        except (urlerror.HTTPError, urlerror.URLError, TimeoutError) as exc:
            raise RuntimeError(f"MinerU query failed: {exc}") from exc

        if query_resp.get('code') != 0:
            raise RuntimeError(
                f"MinerU query failed: {query_resp.get('msg', 'unknown error')}"
            )

        query_data = query_resp.get('data')
        results = []
        if isinstance(query_data, dict):
            results = _api_field(query_data, 'extract_result', 'extract_results')

        if isinstance(results, list) and results:
            entry = results[0] if isinstance(results[0], dict) else {}
            state = entry.get('state')
            if state == 'done':
                candidate = _api_field(entry, 'full_zip_url')
                if isinstance(candidate, str) and candidate:
                    zip_url = candidate
                    break
                raise RuntimeError(
                    "MinerU result is done but no full_zip_url was returned"
                )
            if state == 'failed':
                err_msg = entry.get('err_msg', 'unknown error')
                raise RuntimeError(f"MinerU OCR failed: {err_msg}")

        time.sleep(settings.poll_interval)

    if not zip_url:
        raise RuntimeError("MinerU OCR timed out waiting for parsing result")

    try:
        payload = _http_get_bytes(zip_url, timeout=request_timeout)
        log.debug("Downloaded MinerU result zip (%d bytes)", len(payload))
        return payload
    except (urlerror.HTTPError, urlerror.URLError, TimeoutError) as exc:
        raise RuntimeError(f"MinerU download failed: {exc}") from exc


def _load_pages_from_zip(
    zip_bytes: bytes,
    *,
    include_discarded: bool,
    fallback_width: float,
    fallback_height: float,
    fallback_dpi: float,
) -> list[tuple[OcrElement, str]]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = archive.namelist()

        layout_name = next(
            (name for name in names if name.endswith('layout.json')), None
        )
        if layout_name:
            layout_data = json.loads(archive.read(layout_name).decode('utf-8'))
            if isinstance(layout_data, dict):
                pages = _parse_layout_result(
                    layout_data, include_discarded=include_discarded
                )
                if pages:
                    return pages

        content_candidates = [
            name
            for name in names
            if name.endswith('_content_list.json')
            or name.endswith('content_list_v2.json')
        ]
        for content_name in content_candidates:
            content_data = json.loads(archive.read(content_name).decode('utf-8'))
            pages = _parse_content_list_result(
                content_data,
                fallback_width=fallback_width,
                fallback_height=fallback_height,
                fallback_dpi=fallback_dpi,
            )
            if pages:
                return pages
    return []


def _build_settings(options: OcrOptions) -> MinerUSettings:
    token = str(getattr(options, 'mineru_api_token', '') or '').strip()
    base_url = str(getattr(options, 'mineru_base_url', _DEFAULT_BASE_URL)).strip()
    model_version = str(
        getattr(options, 'mineru_model_version', _DEFAULT_MODEL)
    ).strip() or _DEFAULT_MODEL
    language = str(getattr(options, 'mineru_language', _DEFAULT_LANGUAGE)).strip()
    if not language:
        language = _DEFAULT_LANGUAGE

    return MinerUSettings(
        api_token=token,
        base_url=base_url or _DEFAULT_BASE_URL,
        model_version=model_version,
        language=language,
        enable_formula=bool(getattr(options, 'mineru_enable_formula', True)),
        enable_table=bool(getattr(options, 'mineru_enable_table', True)),
        poll_interval=float(getattr(options, 'mineru_poll_interval', 5.0)),
        timeout=float(getattr(options, 'mineru_timeout', 600.0)),
        include_discarded=bool(getattr(options, 'mineru_include_discarded', False)),
    )


class MinerUOcrEngine(OcrEngine):
    """Implements OCR with MinerU API."""

    @staticmethod
    def version() -> str:
        return "API"

    @staticmethod
    def creator_tag(options: OcrOptions) -> str:
        return "OCRmyPDF fpdf2 + MinerU OCR API"

    def __str__(self) -> str:
        return "MinerU OCR API"

    @staticmethod
    def languages(options: OcrOptions) -> set[str]:
        if options.languages:
            return set(options.languages)
        return {"eng"}

    @staticmethod
    def get_orientation(
        input_file: Path, options: OcrOptions
    ) -> OrientationConfidence:
        del input_file, options
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options: OcrOptions) -> float:
        del input_file, options
        return 0.0

    @staticmethod
    def supports_generate_ocr() -> bool:
        return True

    @staticmethod
    def generate_ocr(
        input_file: Path,
        options: OcrOptions,
        page_number: int = 0,
    ) -> tuple[OcrElement, str]:
        settings = _build_settings(options)

        with Image.open(input_file) as image:
            width, height = image.size
            dpi_raw = image.info.get('dpi', (72, 72))
            if isinstance(dpi_raw, list | tuple) and dpi_raw:
                image_dpi = float(dpi_raw[0])
            else:
                image_dpi = float(dpi_raw) if dpi_raw else 72.0

        if image_dpi <= 0:
            image_dpi = 72.0

        zip_bytes = _run_mineru_api(settings, input_file)
        pages = _load_pages_from_zip(
            zip_bytes,
            include_discarded=settings.include_discarded,
            fallback_width=float(width),
            fallback_height=float(height),
            fallback_dpi=image_dpi,
        )

        if not pages:
            empty_page = OcrElement(
                ocr_class=OcrClass.PAGE,
                bbox=BoundingBox(
                    left=0,
                    top=0,
                    right=float(width),
                    bottom=float(height),
                ),
                dpi=image_dpi,
                page_number=page_number,
            )
            return empty_page, ""

        selected_index = page_number if 0 <= page_number < len(pages) else 0
        page, text = pages[selected_index]
        page.page_number = page_number
        return page, text

    @staticmethod
    def generate_hocr(
        input_file: Path, output_hocr: Path, output_text: Path, options: OcrOptions
    ) -> None:
        del input_file, output_hocr, output_text, options
        raise NotImplementedError(
            "MinerUOcrEngine does not support hOCR output. Use --pdf-renderer fpdf2."
        )

    @staticmethod
    def generate_pdf(
        input_file: Path, output_pdf: Path, output_text: Path, options: OcrOptions
    ) -> None:
        del input_file, output_pdf, output_text, options
        raise NotImplementedError(
            "MinerUOcrEngine cannot generate sandwich renderer PDFs. "
            "Use --pdf-renderer fpdf2."
        )


def _validate_mineru_settings(settings: MinerUSettings) -> None:
    if not settings.api_token:
        raise BadArgsError(
            "MinerU OCR requires an API token. Set --mineru-api-token or "
            "MINERU_API_TOKEN."
        )
    if settings.poll_interval <= 0:
        raise BadArgsError("--mineru-poll-interval must be greater than 0")
    if settings.timeout <= 0:
        raise BadArgsError("--mineru-timeout must be greater than 0")
    if settings.model_version not in {"pipeline", "vlm", "MinerU-HTML"}:
        raise BadArgsError(
            "--mineru-model-version must be one of: pipeline, vlm, MinerU-HTML"
        )


@hookimpl
def add_options(parser):
    mineru = parser.add_argument_group("MinerU", "Options for MinerU OCR API")
    mineru.add_argument(
        '--mineru-api-token',
        dest='mineru_api_token',
        default=os.environ.get('MINERU_API_TOKEN', ''),
        metavar='TOKEN',
        help="MinerU API token. Can also be provided through MINERU_API_TOKEN.",
    )
    mineru.add_argument(
        '--mineru-base-url',
        dest='mineru_base_url',
        default=os.environ.get('MINERU_BASE_URL', _DEFAULT_BASE_URL),
        metavar='URL',
        help=f"MinerU API base URL (default: {_DEFAULT_BASE_URL}).",
    )
    mineru.add_argument(
        '--mineru-model-version',
        dest='mineru_model_version',
        choices=['pipeline', 'vlm', 'MinerU-HTML'],
        default=_DEFAULT_MODEL,
        help="MinerU model version to use.",
    )
    mineru.add_argument(
        '--mineru-language',
        dest='mineru_language',
        default=_DEFAULT_LANGUAGE,
        metavar='LANG',
        help="MinerU language code parameter (default: ch).",
    )
    mineru.add_argument(
        '--mineru-enable-formula',
        action=argparse.BooleanOptionalAction,
        dest='mineru_enable_formula',
        default=True,
        help="Enable MinerU formula recognition.",
    )
    mineru.add_argument(
        '--mineru-enable-table',
        action=argparse.BooleanOptionalAction,
        dest='mineru_enable_table',
        default=True,
        help="Enable MinerU table recognition.",
    )
    mineru.add_argument(
        '--mineru-poll-interval',
        type=numeric(float, 0.1),
        default=5.0,
        metavar='SECONDS',
        dest='mineru_poll_interval',
        help="Polling interval for MinerU task status.",
    )
    mineru.add_argument(
        '--mineru-timeout',
        type=numeric(float, 1.0),
        default=600.0,
        metavar='SECONDS',
        dest='mineru_timeout',
        help="Maximum wait time for MinerU OCR task.",
    )
    mineru.add_argument(
        '--mineru-include-discarded',
        action=argparse.BooleanOptionalAction,
        default=False,
        dest='mineru_include_discarded',
        help="Include discarded blocks such as headers and page numbers in OCR output.",
    )


@hookimpl
def check_options(options):
    if options.ocr_engine != 'mineru':
        return
    if options.pdf_renderer not in ('auto', 'fpdf2'):
        raise BadArgsError(
            "MinerU OCR requires --pdf-renderer auto or --pdf-renderer fpdf2."
        )
    settings = _build_settings(options)
    _validate_mineru_settings(settings)

    jobs = options.jobs or 0
    if jobs > 1:
        log.warning(
            "MinerU OCR is a remote API and may rate-limit concurrent uploads. "
            "Consider using --jobs 1 if requests fail."
        )


@hookimpl
def get_ocr_engine(options):
    if options is not None:
        ocr_engine = getattr(options, 'ocr_engine', 'auto')
        if ocr_engine != 'mineru':
            return None
    return MinerUOcrEngine()
