RNA_2Y_BEFORE_CV_GENES: set[str] = set(
    [
        "CAMK2N2",
        "AC093826.2",
        "LACC1",
        "CSF2",
        "HNRNPA1P9",
        "AL445235.1",
        "AC025580.2",
        "AC025198.1",
        "AC093525.8",
        "OR52N5",
        "RBMXL3",
        "AC004241.5",
        "AC138647.1",
        "AL449283.1",
        "AC130366.1",
        "AC063947.2",
        "H19",
        "SGSM1",
        "HIGD2B",
        "ZMYND12",
    ]
)

PREDEFINED_HCC_2Y_CV_GENES: set[str] = set(
    [
        "CFH",
        "AP2B1",
        "REX1BD",
        "MYCBP2",
        "ABCB4",
        "PON1",
        "ACSM3",
        "MCUB",
        "CALCR",
        "SLC25A13",
        "PDK4",
        "ARF5",
        "SLC7A2",
        "ALS2",
        "M6PR",
        "AOC1",
        "RAD52",
        "CYP51A1",
        "RALA",
        "USH1C",
    ]
)

GENE_SET = RNA_2Y_BEFORE_CV_GENES.union(PREDEFINED_HCC_2Y_CV_GENES)
