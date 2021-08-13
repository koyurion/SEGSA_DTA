

standard_aa_names = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]


aa1 = "ACDEFGHIKLMNPQRSTVWY"
aa3 = standard_aa_names

d3_to_d1 = {}
d1_to_d3 = {}

# Create some lookup tables
for i in range(0, 20):
    n1 = aa1[i]
    n3 = aa3[i]
    d3_to_d1[n3] = n1
    d1_to_d3[n1] = n3


def three_to_one(s):
    """Three letter code to one letter code. """

    return d3_to_d1[s.upper()]


def one_to_three(s):
    """One letter code to three letter code. """

    return d1_to_d3[s.upper()]

