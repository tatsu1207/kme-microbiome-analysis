"""
ML Analysis Configuration
Integrative model: questionnaire + gut microbiome → metabolic health
"""
from pathlib import Path

# === Paths ===
PROJECT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_DIR / "ml"
OUTPUT_DIR = ML_DIR / "output"

# Data sources — adjust these paths to your environment
# KMEDATA: questionnaire data from KIOM (available upon reasonable request)
KMEDATA_CSV = PROJECT_DIR / "data/kmedata_merged.csv"
# DADA2 output: ASV table and taxonomy from 16S-Pipeline (PRJNA1472377)
DADA2_DIR = PROJECT_DIR / "data/dada2"
ASV_TABLE = DADA2_DIR / "asv_table.tsv"
TAXONOMY = DADA2_DIR / "taxonomy.tsv"
PICRUST2_PATHWAYS = DADA2_DIR / "picrust2/pathways_out/path_abun_unstrat.tsv.gz"
PICRUST2_PATHWAYS_DESC = DADA2_DIR / "picrust2/pathways_out/path_abun_unstrat_described.tsv.gz"

# Mothur OTU-based data (optional, for mothur-based comparison)
MOTHUR_DIR = PROJECT_DIR / "data/mothur"
MOTHUR_SHARED = MOTHUR_DIR / "DS.shared"
MOTHUR_TAXONOMY = MOTHUR_DIR / "DS.taxonomy"

# === Feature selection ===
# Top N features by variance for each microbiome block
TOP_N_GENERA = 50
TOP_N_ASVS = 100
TOP_N_PATHWAYS = 50
MIN_ASV_PREVALENCE = 0.05  # ASV must be present in >=5% of samples

# Minimum sample availability rate
MIN_AVAILABILITY = 0.90

# === Outcome definitions ===
# Outcomes are defined from clinical lab values.
# IMPORTANT: all columns used to define outcomes are EXCLUDED from features
# to prevent data leakage.
OUTCOMES = {
    "obesity": {
        "cols": ["PBF"],
        "threshold_M": 25.0,  # percent body fat
        "threshold_F": 30.0,
        "direction": ">=",
        "label": "Elevated body fat",
    },
    "hypertension": {
        "cols": ["SYSBP", "DIABP"],
        "threshold_SBP": 130.0,
        "threshold_DBP": 85.0,
        "direction": "or",
        "label": "Elevated blood pressure",
    },
    "prediabetes": {
        "cols": ["GLUC"],
        "threshold": 100.0,
        "direction": ">=",
        "label": "Elevated fasting glucose",
    },
    "liver_dysfunction": {
        "cols": ["ALT", "GGT"],
        "threshold_ALT": 40.0,
        "threshold_GGT": 60.0,
        "direction": "or",
        "label": "Liver dysfunction",
    },
}

# Columns used to define outcomes — EXCLUDED from all feature matrices.
# Includes the direct measurements and closely related/derived measurements
# (e.g., left/right BP, InBody fat/muscle, waist circumference).
OUTCOME_LEAKAGE_COLS = {
    # Body composition (defines obesity)
    "PBF", "TBW", "FAT", "FAT_RFMIN", "FAT_RFMAX", "SLM", "SLM_RFMIN", "SLM_RFMAX",
    "RMR", "VE", "WEIGHT", "WEIGHTHEPR",
    "WAISTCIR", "WAISTCIRHEPR", "WAISTCIR3D", "WAISTCIRCD",
    # Blood pressure (defines hypertension)
    "SYSBP", "SYSBPLT", "SYSBPRT", "DIABP", "DIABPLT", "DIABPRT",
    # Blood glucose (defines prediabetes)
    "GLUC", "GLUCCD", "GLUCCDYN", "GLUCYN",
    # Liver enzymes (defines liver dysfunction)
    "AST", "ASTCD", "ASTCDYN", "ASTYN",
    "ALT", "ALTCD", "ALTCDYN", "ALTYN",
    "GGT", "GGTCD", "GGTCDYN", "GGTYN",
    # Other lab values closely tied to metabolic state
    "CREAT", "CREATCD", "CREATCDYN", "CREATYN",
    "GFRE", "GFRECD", "GFRECDYN", "GFREYN",
    "HGB", "HGBCD", "HGBCDYN", "HGBYN",
    # Digestive / stool columns (define digestive outcomes)
    "DIGEST01", "DIGEST02_01", "DIGEST02_02", "DIGEST02_03", "DIGEST02_04",
    "DIGEST02_05", "DIGEST02_06", "DIGEST02_07", "DIGEST02_08", "DIGEST02_09",
    "DIGEST03_01", "DIGEST03_02", "DIGEST04_01", "DIGEST04_02",
    "DIGEST05_01", "DIGEST05_02", "DIGEST06_01", "DIGEST06_02",
    "DIGEST07_01", "DIGEST07_02", "DIGEST08_01", "DIGEST08_02",
    "DIGEST09_01", "DIGEST09_02", "DIGEST10_01", "DIGEST10_02",
    "STOOL01_DAY", "STOOL01_FREQ", "STOOLDAY", "STOOL02", "STOOL03",
    "STOOL04", "STOOL05_01", "STOOL05_02", "STOOL05_03", "STOOL05_04",
    "STOOL05_05", "STOOL05_06", "STOOL06", "STOOL07",
    "STOOL08_01", "STOOL08_02", "STOOL08_03", "STOOL08_04",
    "STOOL08_05", "STOOL08_06", "STOOL08_07", "STOOL08_08",
}

# Clinic-instrument columns — require in-person measurement equipment.
# Excluded from M1 to keep it pure self-report only.
CLINIC_INSTRUMENT_COLS = {
    # 3D body scanner circumferences
    "FOREHEADCIR3D", "NECKCIR3D", "CHESTCIR3D", "RIBCIR3D", "ILIUMCIR3D", "HIPCIR3D",
    # Skin moisture/oil (instrument)
    "FHMOIST", "FHOIL", "BRMOISTLT", "BRMOISTRT", "BROILLT", "BROILRT",
    "ABRMOISTLT", "ABRMOISTRT", "ABROILLT", "ABROILRT",
    # Tongue diagnosis (clinician)
    "LIGHTRED", "COATED", "TOOTHMARK",
    # Body temperature (thermography)
    "TEMP_01", "TEMP_02", "TEMP_03",
    # Face anthropometry (3D scanner)
    "FRCORCOIDHTLT", "FRCORCOIDHTRT", "FRCORCOIDWDLT", "FRCORCOIDWDRT",
    "FREARHTLT", "FREARHTRT", "FREARWDLT", "FREARWDRT",
    "FRASISHTLT", "FRASISHTRT", "FRASISWDLT", "FRASISWDRT",
    # Back/shoulder (3D scanner)
    "BKSHLDRDT", "BKSHLDRDTDIR",
    # Sasang constitution measured by clinician (not self-report questionnaire)
    "SCATNMEAS_BODY", "SCATNMEAS_FACE_FRONT", "SCATNMEAS_FACE_LATERAL",
    "SCATNMEAS_VOICE", "SCATNMEAS_PERSONALITY",
    "ATNMEASCC", "ATNMEASPS", "ATNMEAS_FRONT", "ATNMEAS_BACK",
    "SCATPERSON01", "SCATPERSON02", "SCATPERSON03", "SCATPERSON04",
    "SCATPERSON05", "SCATPERSON06", "SCATPERSON07", "SCATPERSON08",
    "SCATPERSON09", "SCATPERSON10", "SCATPERSON11", "SCATPERSON12",
    # Pulse diagnosis (instrument)
    "PULSELT", "PULSERT", "PULSERLT", "PULSERRT",
    "HDNMEAS_PULSELT", "HDNMEAS_PULSERT",
    # HRV (heart rate variability instrument)
    "TP", "VLF", "LFHF", "TPTSCR", "HFTSCR", "LFTSCR",
    # Saliva test
    "SALIVATE",
    # Abdominal palpation (clinician)
    "ABNMEAS_REFDLTD",
    # Visual acuity / hearing (instruments)
    "VALT", "VART",
    # Urinalysis (lab)
    "URPROTCD",
    # X-ray
    "XRAYCD",
    # Height measured at clinic (self-reported HEIGHT is kept)
    "HEIGHTHEPR",
}

# === Questionnaire feature blocks (non-invasive only) ===
# These are what a subject can report WITHOUT lab tests.
QUESTIONNAIRE_BLOCKS = {
    "demographics": {
        "cols": ["AGE", "SEX", "HEIGHT", "EDLEVEL", "MARISTAT", "EMPJOB", "HANDDOM"],
        "description": "Age, sex, height, socioeconomic",
    },
    "lifestyle": {
        "prefix": ["CIGAR", "ALCOHOL", "IPAQ"],
        "description": "Smoking, alcohol, physical activity",
    },
    "sleep_stress": {
        "prefix": ["PSQI", "PSS", "SLEEP"],
        "description": "Sleep quality (PSQI), perceived stress (PSS)",
    },
    "emotion": {
        "prefix": ["CSEIS"],
        "description": "Emotional state inventory (7 subscales)",
    },
    "urination_eating": {
        "prefix": ["URINE", "EAT"],
        "description": "Urination habits, eating habits",
    },
    "allergy": {
        "prefix": ["ALLG", "ATPHY"],
        "description": "Allergy history",
    },
    "disease_history": {
        "prefix": ["MHHTN", "MHDM", "MHDLP"],
        "description": "Self-reported chronic disease history",
    },
    "korean_medicine": {
        "prefix": ["SCAT", "CHQ", "CHPIQ", "SDFI", "SUI", "SPQ"],
        "description": "Sasang constitution, cold-heat questionnaires",
    },
    "sweat_temp": {
        "prefix": ["SWEAT", "CHQ"],
        "description": "Perspiration, cold-heat patterns",
    },
}

# === Model settings ===
OUTER_CV_FOLDS = 5
INNER_CV_FOLDS = 5
RANDOM_STATE = 42
