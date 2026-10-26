"""Read authors' feature labels without importing factor returns or t-statistics."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile

NAMESPACE = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SOURCE_URL = "https://raw.githubusercontent.com/bkelly-lab/jkp-data/main/src/jkp/data/resources/factor_details.xlsx"


def characteristic_definitions(path):
    """Project only identifier, economic label and literature attribution.

    Published directions and in-sample performance statistics are deliberately
    excluded: the workbook documents variables, not evidence for our sample.
    """
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(t.text or "" for t in cell.findall(".//m:t", NAMESPACE))
                for cell in ET.fromstring(
                    archive.read("xl/sharedStrings.xml")
                ).findall("m:si", NAMESPACE)
            ]
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        definitions = {}
        for row in root.findall("m:sheetData/m:row", NAMESPACE)[1:]:
            fields = {}
            for cell in row.findall("m:c", NAMESPACE):
                column = re.match(r"[A-Z]+", cell.attrib.get("r", "")).group()
                if column not in {"C", "D", "G"}:
                    continue
                value = cell.find("m:v", NAMESPACE)
                text = (
                    value.text
                    if value is not None
                    else "".join(
                        t.text or "" for t in cell.findall(".//m:t", NAMESPACE)
                    )
                )
                fields[column] = (
                    strings[int(text)] if cell.attrib.get("t") == "s" else text
                )
            identifier = fields.get("D")
            if identifier and re.fullmatch(r"[a-z][a-z0-9_]*", identifier):
                definitions[identifier] = {
                    "meaning": fields.get("C", ""),
                    "reference": fields.get("G", ""),
                }
        return definitions
