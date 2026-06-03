# utils.py

import re

import fitz


def load_pdf(path):

    doc = fitz.open(path)

    pages = []

    for page in doc:
        pages.append(
            page.get_text("text")
        )

    return "\n".join(pages)


def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"references.*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def audit_chunk(chunk):

    issues = []

    if len(chunk.split()) < 40:
        issues.append("too_short")

    if chunk.count(".") < 2:
        issues.append("low_information")

    if "references" in chunk.lower():
        issues.append("references")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }