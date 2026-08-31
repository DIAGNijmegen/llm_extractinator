"""Generates the synthetic datasets under ``devtests/data``.

Kept in the repo so the data can be regenerated or extended, but the generated
JSON is committed: a dev harness has to be reproducible, and re-rolling the
inputs between two runs would make the comparison meaningless.

Everything here is invented. No real report, patient or identifier appears.
"""

import json
import random
from pathlib import Path

DATA = Path(__file__).parent / "data"

# ── short reports, English ────────────────────────────────────────────

REPORTS_EN = [
    "CT of the abdomen with intravenous contrast. Comparison: CT of 14 March. "
    "The liver is normal in size and contour. A 12 mm hypodense lesion in "
    "segment VI is unchanged and remains consistent with a simple cyst. The "
    "spleen, pancreas and adrenal glands are unremarkable. No free fluid. "
    "IMPRESSION: Stable simple hepatic cyst. No acute abdominal pathology. "
    "Recommendation: no further imaging required.",

    "MRI of the left knee without contrast. There is a full-thickness tear of "
    "the anterior cruciate ligament with associated bone bruising of the "
    "lateral femoral condyle measuring 22 mm. The medial meniscus shows a "
    "grade 2 signal without a definite tear. Moderate joint effusion. "
    "IMPRESSION: Acute complete ACL rupture. Orthopaedic referral advised "
    "within two weeks.",

    "Chest radiograph, PA and lateral. The lungs are clear. Cardiac silhouette "
    "is within normal limits. No pleural effusion or pneumothorax. Bony "
    "thorax intact. IMPRESSION: Normal chest radiograph.",

    "CT of the chest with contrast. Comparison: CT of 2 February. A spiculated "
    "nodule in the right upper lobe measures 18 mm, previously 14 mm. Two sub-"
    "centimetre nodules in the left lower lobe are unchanged. Mediastinal "
    "lymph nodes up to 11 mm in the short axis. No pleural effusion. "
    "IMPRESSION: Interval growth of the right upper lobe nodule, suspicious "
    "for malignancy. Urgent multidisciplinary discussion recommended.",

    "Ultrasound of the right upper quadrant. The gallbladder contains several "
    "mobile echogenic foci with posterior acoustic shadowing, the largest 9 mm. "
    "The wall measures 2 mm and is not thickened. No pericholecystic fluid. "
    "The common bile duct measures 4 mm. IMPRESSION: Cholelithiasis without "
    "sonographic evidence of acute cholecystitis.",

    "MRI of the brain with and without contrast. Multiple T2 hyperintense "
    "lesions in the periventricular and juxtacortical white matter. One lesion "
    "in the left corona radiata demonstrates incomplete ring enhancement. No "
    "mass effect or midline shift. IMPRESSION: Findings compatible with a "
    "demyelinating process; one actively enhancing lesion. Neurology review "
    "recommended.",

    "Plain radiograph of the right wrist, two views. There is a transverse "
    "fracture through the distal radial metaphysis with 15 degrees of dorsal "
    "angulation. The distal radioulnar joint is congruent. No scaphoid "
    "fracture identified. IMPRESSION: Acute distal radius fracture. "
    "Recommendation: closed reduction and cast immobilisation.",

    "CT of the head without contrast following a fall. No intracranial "
    "haemorrhage, mass effect or midline shift. The ventricles are of normal "
    "size. No skull fracture. Mild chronic small vessel ischaemic change in "
    "the deep white matter bilaterally. IMPRESSION: No acute intracranial "
    "injury.",

    "MRI of the lumbar spine without contrast. At L4-L5 there is a broad-based "
    "posterior disc protrusion causing moderate central canal stenosis and "
    "bilateral lateral recess narrowing. At L5-S1 a right paracentral "
    "protrusion contacts the traversing right S1 nerve root. Facet arthropathy "
    "at both levels. IMPRESSION: Multilevel degenerative change; right S1 "
    "nerve root contact at L5-S1 correlating with the reported symptoms.",

    "Mammography, bilateral, with tomosynthesis. Breast density category C. In "
    "the upper outer quadrant of the left breast there is a 9 mm irregular "
    "mass with associated pleomorphic microcalcifications. No axillary "
    "lymphadenopathy. The right breast is unremarkable. IMPRESSION: Suspicious "
    "left breast mass, BI-RADS 4. Recommendation: ultrasound-guided core "
    "biopsy.",
]

# ── short reports, Dutch ──────────────────────────────────────────────
# The same clinical shapes, written in Dutch rather than translated word for
# word: the point is realistic Dutch phrasing, because that is where the
# cl100k_base token estimate drifts most.

REPORTS_NL = [
    "CT abdomen met intraveneus contrast. Vergelijking: CT van 14 maart. De "
    "lever is normaal van grootte en begrenzing. Een hypodense laesie van 12 mm "
    "in segment VI is ongewijzigd en past bij een simpele cyste. Milt, pancreas "
    "en bijnieren zijn niet afwijkend. Geen vrij vocht. CONCLUSIE: Stabiele "
    "simpele levercyste. Geen acute abdominale pathologie. Geen vervolgonderzoek "
    "geïndiceerd.",

    "MRI linkerknie zonder contrast. Er is een volledige ruptuur van de voorste "
    "kruisband met bijbehorende botcontusie van de laterale femurcondyl over "
    "22 mm. De mediale meniscus toont een graad 2 signaal zonder duidelijke "
    "scheur. Matige gewrichtseffusie. CONCLUSIE: Acute complete VKB-ruptuur. "
    "Verwijzing orthopedie binnen twee weken geadviseerd.",

    "Thoraxfoto, PA en lateraal. De longen zijn schoon. Hartcontour binnen "
    "normale grenzen. Geen pleuravocht of pneumothorax. Benige thorax intact. "
    "CONCLUSIE: Normale thoraxfoto.",

    "CT thorax met contrast. Vergelijking: CT van 2 februari. Een spiculaire "
    "nodus in de rechter bovenkwab meet 18 mm, voorheen 14 mm. Twee subcentimeter "
    "noduli in de linker onderkwab zijn ongewijzigd. Mediastinale lymfeklieren "
    "tot 11 mm in de korte as. Geen pleuravocht. CONCLUSIE: Intervalgroei van de "
    "nodus in de rechter bovenkwab, verdacht voor maligniteit. Spoed MDO "
    "geadviseerd.",

    "Echografie rechter bovenbuik. De galblaas bevat meerdere mobiele echogene "
    "foci met dorsale schaduw, de grootste 9 mm. De wand meet 2 mm en is niet "
    "verdikt. Geen pericholecystisch vocht. De ductus choledochus meet 4 mm. "
    "CONCLUSIE: Cholelithiasis zonder echografische aanwijzingen voor acute "
    "cholecystitis.",

    "MRI cerebrum met en zonder contrast. Multipele T2-hyperintense laesies "
    "periventriculair en juxtacorticaal. Eén laesie in de linker corona radiata "
    "toont incomplete ringaankleuring. Geen massawerking of middellijnverplaatsing. "
    "CONCLUSIE: Beeld passend bij een demyeliniserend proces; één actief "
    "aankleurende laesie. Beoordeling neurologie geadviseerd.",

    "Röntgenfoto rechterpols, twee richtingen. Er is een transversale fractuur "
    "door de distale radiusmetafyse met 15 graden dorsale angulatie. Het distale "
    "radioulnaire gewricht is congruent. Geen scafoïdfractuur zichtbaar. "
    "CONCLUSIE: Acute distale radiusfractuur. Advies: gesloten repositie en "
    "gipsimmobilisatie.",

    "CT cerebrum zonder contrast na een val. Geen intracraniële bloeding, "
    "massawerking of middellijnverplaatsing. De ventrikels zijn normaal van "
    "grootte. Geen schedelfractuur. Milde chronische kleinevatenziekte in de "
    "diepe witte stof beiderzijds. CONCLUSIE: Geen acuut intracranieel letsel.",

    "MRI lumbale wervelkolom zonder contrast. Op niveau L4-L5 een breedbasige "
    "posterieure discusprotrusie met matige centrale kanaalstenose en bilaterale "
    "vernauwing van de laterale recessus. Op L5-S1 een rechts paracentrale "
    "protrusie met contact met de doorlopende rechter S1-wortel. Facetartrose op "
    "beide niveaus. CONCLUSIE: Multilevel degeneratieve afwijkingen; contact met "
    "de rechter S1-wortel op L5-S1 correleert met de gemelde klachten.",

    "Mammografie, bilateraal, met tomosynthese. Densiteitscategorie C. In het "
    "laterale bovenkwadrant van de linkerborst een irregulaire massa van 9 mm met "
    "bijbehorende pleomorfe microcalcificaties. Geen okselklierpathologie. De "
    "rechterborst is niet afwijkend. CONCLUSIE: Verdachte massa linkerborst, "
    "BI-RADS 4. Advies: echogeleide dikke-naaldbiopsie.",
]

# ── long reports ──────────────────────────────────────────────────────
# Assembled from hand-written sections so each is genuinely different in
# content rather than one paragraph repeated, while still reaching the couple
# of thousand tokens that make context sizing interesting.

SECTIONS = [
    "CLINICAL HISTORY: {age}-year-old presenting with {complaint} of {duration} "
    "duration. Referred from {source} for further characterisation. Relevant "
    "past history includes {history}.",
    "TECHNIQUE: {modality} performed on a {vendor} scanner. {contrast} "
    "Reconstructions were obtained in the axial, coronal and sagittal planes at "
    "{slice} mm slice thickness. Image quality was {quality}.",
    "COMPARISON: Prior {modality} of {date}, and the corresponding report, were "
    "available for direct comparison. Where measurements are given below they "
    "have been made on comparable sequences at the same anatomical level.",
    "LUNGS AND AIRWAYS: The trachea and central bronchi are patent to the "
    "segmental level. There is {lung_finding} in the {lobe}. No consolidation, "
    "cavitation or bronchiectasis elsewhere. The pulmonary vasculature is "
    "unremarkable in calibre and distribution.",
    "MEDIASTINUM: The heart is normal in size. No pericardial effusion. "
    "Mediastinal lymph nodes measure up to {node} mm in the short axis, within "
    "normal limits for size. The thoracic aorta is of normal calibre without "
    "dissection or aneurysm. The oesophagus is unremarkable where visualised.",
    "LIVER AND BILIARY: The liver measures {liver} cm in craniocaudal extent "
    "and is homogeneous in attenuation. A {lesion} mm {lesion_type} is noted in "
    "segment {segment}, unchanged from the prior study. The gallbladder is "
    "distended without wall thickening or pericholecystic fluid. The intra- and "
    "extrahepatic bile ducts are not dilated.",
    "PANCREAS, SPLEEN AND ADRENALS: The pancreas is normal in size and contour "
    "without ductal dilatation or peripancreatic stranding. The spleen measures "
    "{spleen} cm and is homogeneous. Both adrenal glands are normal in "
    "thickness without discrete nodule.",
    "KIDNEYS AND URINARY TRACT: Both kidneys enhance symmetrically. A {cyst} mm "
    "simple cortical cyst is present in the {side} kidney, of no clinical "
    "significance. No hydronephrosis, calculus or perinephric stranding. The "
    "ureters are not dilated. The bladder is partially distended with a smooth "
    "wall.",
    "BOWEL AND PERITONEUM: The stomach and small bowel are non-dilated. No "
    "free intraperitoneal gas or fluid. The appendix is not confidently "
    "identified; there is no periappendiceal inflammatory change. Colonic "
    "diverticulosis is noted in the sigmoid without evidence of acute "
    "diverticulitis.",
    "MUSCULOSKELETAL: Degenerative change is present in the {spine_level} "
    "spine with anterior osteophytes and facet hypertrophy. No aggressive or "
    "lytic bone lesion. No acute fracture. Muscle bulk and fat planes are "
    "preserved.",
    "INCIDENTAL FINDINGS: {incidental} This was not present on the prior study "
    "and is felt to be of doubtful clinical significance in isolation, though "
    "correlation with the clinical picture is advised.",
    "IMPRESSION: 1. {impression_one}. 2. {impression_two}. 3. No acute finding "
    "requiring immediate intervention.",
    "RECOMMENDATION: {recommendation} These findings were discussed with the "
    "referring clinician at the time of reporting.",
]

FILL = {
    "age": ["54", "61", "38", "72", "47"],
    "complaint": [
        "intermittent right upper quadrant pain",
        "unexplained weight loss and fatigue",
        "a persistent cough productive of clear sputum",
        "shortness of breath on exertion",
    ],
    "duration": ["three weeks", "four months", "several days", "over a year"],
    "source": ["the outpatient clinic", "the emergency department", "primary care"],
    "history": [
        "hypertension and type 2 diabetes",
        "a previous cholecystectomy",
        "no significant comorbidity",
        "a 30 pack-year smoking history",
    ],
    "modality": ["CT of the chest and abdomen", "CT of the abdomen and pelvis"],
    "vendor": ["64-slice", "128-slice", "dual-source"],
    "contrast": [
        "Intravenous contrast was administered in the portal venous phase.",
        "No intravenous contrast was administered owing to impaired renal function.",
        "Intravenous contrast was administered in arterial and portal venous phases.",
    ],
    "slice": ["1.0", "1.25", "2.0"],
    "quality": ["diagnostic", "limited by respiratory motion but diagnostic"],
    "date": ["11 January", "28 September", "3 June"],
    "lung_finding": [
        "a 7 mm subpleural nodule",
        "mild dependent atelectasis",
        "a small area of ground-glass opacity",
    ],
    "lobe": ["right lower lobe", "left upper lobe", "lingula"],
    "node": ["8", "9", "11"],
    "liver": ["15.2", "16.8", "14.1"],
    "lesion": ["12", "8", "19"],
    "lesion_type": ["simple cyst", "haemangioma", "hypodense lesion"],
    "segment": ["IV", "VI", "VIII"],
    "spleen": ["10.4", "11.9", "9.8"],
    "cyst": ["14", "22", "9"],
    "side": ["left", "right"],
    "spine_level": ["lower thoracic", "lumbar"],
    "incidental": [
        "A small hiatus hernia is noted.",
        "There is a 4 mm gallstone within the gallbladder neck.",
        "Mild splenomegaly is present.",
    ],
    "impression_one": [
        "Stable benign hepatic lesion, unchanged from the prior study",
        "No evidence of metastatic disease within the imaged volume",
        "Interval resolution of the previously described inflammatory change",
    ],
    "impression_two": [
        "Colonic diverticulosis without acute inflammation",
        "Degenerative spinal change consistent with age",
        "Simple renal cyst of no clinical significance",
    ],
    "recommendation": [
        "No further imaging is required on the basis of these findings.",
        "Suggest interval follow-up imaging in twelve months.",
        "Correlation with laboratory results is advised.",
    ],
}


# What actually makes a real staging report long: an enumerated list of every
# measured lesion, each with a location, a size and a comparison. Generating
# these is what takes the long reports from a few hundred tokens to a few
# thousand, which is the range where context sizing becomes interesting.
LESION_SITES = [
    "right hepatic lobe, segment {seg}", "left hepatic lobe, segment {seg}",
    "right lower lobe of the lung", "left upper lobe of the lung",
    "retroperitoneum, para-aortic station", "left adrenal gland",
    "mesenteric root", "right iliac chain", "left supraclavicular fossa",
    "peritoneal reflection, right paracolic gutter", "spleen, upper pole",
    "T{seg} vertebral body", "right acetabulum", "gastrohepatic ligament",
    "porta hepatis", "left external iliac chain",
]
LESION_CHANGE = [
    "previously {prev} mm, representing interval increase",
    "previously {prev} mm, representing interval decrease",
    "unchanged from the prior study",
    "newly apparent since the prior study",
    "previously {prev} mm, stable within measurement error",
]


def _lesion_table(rng: random.Random) -> str:
    lines = ["MEASURABLE DISEASE: Lesions are numbered consistently with the "
             "prior report where correspondence could be established."]
    for index in range(1, rng.randint(16, 26)):
        site = rng.choice(LESION_SITES).format(seg=rng.choice(
            ["II", "III", "IV", "V", "VI", "VII", "VIII", "9", "10", "11"]))
        size = rng.randint(6, 84)
        change = rng.choice(LESION_CHANGE).format(prev=max(4, size + rng.randint(-20, 20)))
        lines.append(
            f"  {index}. Lesion in the {site}, measuring {size} mm in the "
            f"greatest axial dimension ({change}). Margins are "
            f"{rng.choice(['well defined', 'ill defined', 'lobulated', 'spiculated'])}; "
            f"attenuation is {rng.choice(['homogeneous', 'heterogeneous with central necrosis', 'peripherally enhancing'])}."
        )
    return "\n".join(lines)


def _long_report(rng: random.Random) -> str:
    chosen = SECTIONS[:3] + rng.sample(SECTIONS[3:11], 6)
    filled = []
    for section in chosen:
        values = {key: rng.choice(options) for key, options in FILL.items()}
        filled.append(section.format(**values))
    filled.append(_lesion_table(rng))
    for section in SECTIONS[11:]:
        values = {key: rng.choice(options) for key, options in FILL.items()}
        filled.append(section.format(**values))
    return "\n\n".join(filled)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260828)

    (DATA / "reports_en.json").write_text(
        json.dumps([{"text": t} for t in REPORTS_EN], indent=2), encoding="utf-8"
    )
    (DATA / "reports_nl.json").write_text(
        json.dumps([{"text": t} for t in REPORTS_NL], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    long_reports = [{"text": _long_report(rng)} for _ in range(4)]
    (DATA / "reports_long.json").write_text(
        json.dumps(long_reports, indent=2), encoding="utf-8"
    )

    for name in ("reports_en.json", "reports_nl.json", "reports_long.json"):
        rows = json.loads((DATA / name).read_text(encoding="utf-8"))
        chars = [len(r["text"]) for r in rows]
        print(
            f"{name:20s} {len(rows):3d} rows, "
            f"{min(chars):5d}-{max(chars):5d} chars "
            f"(~{min(chars)//4}-{max(chars)//4} tokens)"
        )


if __name__ == "__main__":
    main()
