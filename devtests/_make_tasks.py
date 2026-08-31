"""Generates the task files under ``devtests/tasks``.

Each task exists to probe something the offline suite structurally *cannot*
answer, because a fake model always returns whatever it was told to. Anything a
faked model could verify belongs in ``tests/``, not here.

Schemas are declared inline rather than in parser files so that each probe is
one readable file, and so the inline-dict branch of ``resolve_parser_model`` gets
exercised too.
"""

import json
from pathlib import Path

TASKS = Path(__file__).parent / "tasks"


def field(kind, description, literals=None, optional=False, items=None):
    spec = {"type": kind}
    if description is not None:
        spec["description"] = description
    if literals:
        spec["literals"] = literals
    if optional:
        spec["optional"] = True
    if items:
        spec["items"] = items
    return spec


# The pair used for the description A/B: identical field names and types, one
# set documented and one set bare. Anything else differing between them would
# confound the comparison.
AB_FIELDS = {
    "modality": "The imaging modality used — CT, MRI, ultrasound, radiograph or mammography",
    "body_region": "The anatomical region examined, as named in the report",
    "contrast_used": "Whether intravenous contrast was given, as stated in the technique",
    "primary_finding": "The single most clinically important finding, in the report's own words",
    "measurement_mm": "The size in millimetres of the main lesion, if one is measured",
    "recommendation": "The follow-up action the report recommends, if any",
}

TASK_FILES = {
    "Task101_basic": {
        "Description": (
            "Extract the key facts from a radiology report. The reports are "
            "free text written by radiologists and follow no fixed template."
        ),
        "Data_Path": "reports_en.json",
        "Input_Field": "text",
        "Parser_Format": {
            "modality": field("str", "The imaging modality used, e.g. CT, MRI, ultrasound, radiograph"),
            "body_region": field("str", "The anatomical region imaged, as named in the report"),
            "primary_finding": field("str", "The single most important finding, in the report's own words"),
        },
    },
    "Task102_enums": {
        "Description": (
            "Classify a radiology report along three fixed axes. Choose the "
            "closest option; do not invent new categories."
        ),
        "Data_Path": "reports_en.json",
        "Input_Field": "text",
        "Parser_Format": {
            "severity": field(
                "str", "Overall severity of the findings",
                literals=["normal", "mild", "moderate", "severe"],
            ),
            "urgency": field(
                "str", "How quickly the report implies action is needed",
                literals=["routine", "expedited", "urgent"],
            ),
            "laterality": field(
                "str", "Which side the main finding is on",
                literals=["left", "right", "bilateral", "not_applicable"],
            ),
        },
    },
    "Task103_wide": {
        "Description": (
            "Extract a detailed structured summary of a staging CT report, "
            "covering every organ system the report describes."
        ),
        "Data_Path": "reports_long.json",
        "Input_Field": "text",
        "Parser_Format": {
            "modality": field("str", "The imaging modality and the region covered"),
            "contrast": field("str", "Contrast administration, as described in the technique"),
            "slice_thickness_mm": field("float", "Reconstruction slice thickness in millimetres", optional=True),
            "image_quality": field("str", "The report's own assessment of image quality", optional=True),
            "comparison_date": field("str", "Date of the prior study used for comparison", optional=True),
            "clinical_history": field("str", "The presenting complaint and its duration"),
            "lung_findings": field("str", "Findings in the lungs and airways", optional=True),
            "mediastinum_findings": field("str", "Findings in the mediastinum", optional=True),
            "liver_findings": field("str", "Findings in the liver and biliary system", optional=True),
            "pancreas_findings": field("str", "Findings in the pancreas, spleen and adrenals", optional=True),
            "kidney_findings": field("str", "Findings in the kidneys and urinary tract", optional=True),
            "bowel_findings": field("str", "Findings in the bowel and peritoneum", optional=True),
            "musculoskeletal_findings": field("str", "Musculoskeletal findings", optional=True),
            "largest_lesion_mm": field("int", "Size in millimetres of the largest measured lesion", optional=True),
            "largest_lesion_site": field("str", "Anatomical site of the largest measured lesion", optional=True),
            "lesion_count": field("int", "How many lesions are individually measured", optional=True),
            "new_lesions": field("bool", "Whether any lesion is described as newly apparent", optional=True),
            "growing_lesions": field("bool", "Whether any lesion is described as having increased", optional=True),
            "incidental_finding": field("str", "Any incidental finding noted", optional=True),
            "impression_summary": field("str", "The impression, summarised in one sentence"),
            "recommendation": field("str", "The recommendation given at the end of the report"),
        },
    },
    "Task104_nested": {
        "Description": (
            "Extract every measurement stated in a radiology report, with what "
            "was measured and how large it was."
        ),
        "Data_Path": "reports_en.json",
        "Input_Field": "text",
        "Parser_Format": {
            "measurements": field(
                "list",
                "Every measurement the report states, one entry per measurement",
                items=field(
                    "dict", "A single measurement",
                    items=None,
                ),
            )
        },
    },
    "Task105_optional": {
        "Description": (
            "Extract whatever the report happens to state. Every field is "
            "optional: leave a field out entirely rather than guessing at it."
        ),
        "Data_Path": "reports_en.json",
        "Input_Field": "text",
        "Parser_Format": {
            "modality": field("str", "The imaging modality, if stated", optional=True),
            "primary_finding": field("str", "The main finding, if there is one", optional=True),
            "prior_surgery": field("str", "Any previous surgery the report mentions", optional=True),
            "contrast_reaction": field("str", "Any adverse contrast reaction described", optional=True),
            "radiation_dose": field("str", "The radiation dose, if the report gives one", optional=True),
            "referring_clinician": field("str", "The named referring clinician, if given", optional=True),
        },
    },
    "Task106_reasoning": {
        "Description": (
            "Judge what a radiology report implies. This needs a comparison "
            "against the prior study and an assessment of what should happen "
            "next, not just copying a sentence out."
        ),
        "Data_Path": "reports_en.json",
        "Input_Field": "text",
        "Parser_Format": {
            "change_since_prior": field(
                "str", "How the findings have changed since the prior study",
                literals=["improved", "stable", "worsened", "no_prior_available"],
            ),
            "requires_action": field(
                "bool", "Whether the report implies any clinical action is needed"
            ),
            "rationale": field(
                "str", "One sentence explaining the judgement, citing the report"
            ),
        },
    },
    "Task107_dutch": {
        "Description": (
            "Extract the key facts from a Dutch radiology report. The reports "
            "are free text written by radiologists and follow no fixed template."
        ),
        "Extra_Instructions": (
            "The reports are in Dutch. Give every extracted value in English, "
            "except where the value is a direct quotation from the report."
        ),
        "Data_Path": "reports_nl.json",
        "Input_Field": "text",
        "Parser_Format": {
            "modality": field("str", "The imaging modality used, e.g. CT, MRI, ultrasound, radiograph"),
            "body_region": field("str", "The anatomical region imaged, as named in the report"),
            "primary_finding": field("str", "The single most important finding, in the report's own words"),
        },
    },
    "Task108_long": {
        "Description": (
            "Extract three facts from a long staging CT report. The schema is "
            "deliberately small: what is being exercised is the length of the "
            "input, not the width of the output."
        ),
        "Data_Path": "reports_long.json",
        "Input_Field": "text",
        "Parser_Format": {
            "modality": field("str", "The imaging modality and the region covered"),
            "impression_summary": field("str", "The impression, summarised in one sentence"),
            "recommendation": field("str", "The recommendation given at the end of the report"),
        },
    },
    "Task109_described": {
        "Description": "Extract a structured summary from a radiology report.",
        "Data_Path": "reports_en.json",
        "Input_Field": "text",
        "Parser_Format": {
            name: field(
                "float" if name == "measurement_mm" else
                "bool" if name == "contrast_used" else "str",
                description,
                optional=True,
            )
            for name, description in AB_FIELDS.items()
        },
    },
    "Task110_bare": {
        "Description": "Extract a structured summary from a radiology report.",
        "Data_Path": "reports_en.json",
        "Input_Field": "text",
        "Parser_Format": {
            name: field(
                "float" if name == "measurement_mm" else
                "bool" if name == "contrast_used" else "str",
                None,  # the whole point: no description at all
                optional=True,
            )
            for name in AB_FIELDS
        },
    },
}


def main() -> None:
    TASKS.mkdir(parents=True, exist_ok=True)
    # The nested schema needs its item shape spelled out; the helper above
    # cannot express a nested dict's properties, so it is patched in here.
    TASK_FILES["Task104_nested"]["Parser_Format"]["measurements"]["items"] = {
        "type": "dict",
        "description": "A single measurement",
        "properties": {
            "structure": field("str", "What was measured, e.g. 'hepatic lesion', 'common bile duct'"),
            "size_mm": field("float", "The size in millimetres"),
            "comparison": field(
                "str", "How it compares with the prior study, if stated", optional=True
            ),
        },
    }

    for name, body in TASK_FILES.items():
        path = TASKS / f"{name}.json"
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        fields_count = len(body["Parser_Format"])
        print(f"{name:22s} {body['Data_Path']:20s} {fields_count:2d} field(s)")


if __name__ == "__main__":
    main()
