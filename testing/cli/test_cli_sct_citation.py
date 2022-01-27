import logging
from textwrap import dedent

from spinalcordtoolbox.scripts import sct_citation

logger = logging.getLogger(__name__)


def test_sct_citation(capfd):
    """
    Test the CLI script to ensure that we return the correct citation for BibTex
    """

    sct_citation.main(argv=[])

    out, err = capfd.readouterr()
    assert out == dedent("""\
    @article{DeLeener201724,
    title = "SCT: Spinal Cord Toolbox, an open-source software for processing spinal cord \\{MRI\\} data ",
    journal = "NeuroImage ",
    volume = "145, Part A",
    number = "",
    pages = "24 - 43",
    year = "2017",
    note = "",
    issn = "1053-8119",
    doi = "https://doi.org/10.1016/j.neuroimage.2016.10.009",
    url = "http://www.sciencedirect.com/science/article/pii/S1053811916305560",
    author = "Benjamin De Leener and Simon Lévy and Sara M. Dupont and Vladimir S. Fonov and Nikola Stikov and D. Louis Collins and Virginie Callot and Julien Cohen-Adad",
    keywords = "Spinal cord",
    keywords = "MRI",
    keywords = "Software",
    keywords = "Template",
    keywords = "Atlas",
    keywords = "Open-source ",
    }
    """)
