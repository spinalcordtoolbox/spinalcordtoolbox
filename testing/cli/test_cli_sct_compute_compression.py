import pytest
import logging
import numpy as np
import pandas as pd
import tempfile
import nibabel
import csv

from spinalcordtoolbox.scripts import sct_compute_compression

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def dummy_3d_label_label():
    data = np.zeros([32, 32, 81], dtype=np.uint8)
    data[15, 15, 72] = 1
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.fixture(scope="session")
def dummy_metrics_csv():
    vertlevel = np.empty(38)
    vertlevel.fill(7)
    vertlevel[0:5] = np.ones(5)*8
    vertlevel[-7::] = np.ones(7)*6
    slices = np.arange(44, 82)
    mean_diameter = np.array([6.6862469918429195, 6.55169728444946, 6.389135256069518, 6.304799342877206, 6.041723469159038, 6.439003691181355,
                              6.544616501201308, 6.139186306769116, 6.542551093609961, 6.416558726468434, 6.035723621913116, 5.955694595068289,
                              6.263471044201964, 6.717050230017857, 6.435380314120982, 6.182082168409608, 6.129463897495415, 5.7612477330205305,
                              6.091412318629499, 6.532062074793497, 6.467776708536095, 7.222792805918961, 7.23214311204923, 6.749575410990546,
                              6.62637117360869, 6.424124306698773, 6.820276721945256, 6.745489247204905, 6.423584195632916, 6.370832525207956,
                              6.661876720671968, 6.496059111401223, 6.597853777753301, 6.4723521908617725, 6.534168768380888, 6.125633872831384,
                              6.182984047618388, 6.6483805095089075])
    d = {'Slice (I->S)': slices, 'VertLevel': vertlevel, 'MEAN(diameter_AP)': mean_diameter}
    df = pd.DataFrame(data=d)
    filename = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    df.to_csv(filename)
    return filename


@pytest.fixture(scope="session")
def dummy_metrics_csv_pam50():
    vertlevel = np.empty(51)
    vertlevel.fill(7)
    vertlevel[0:11] = np.ones(11)*8
    vertlevel[-8::] = np.ones(8)*6
    slices = np.arange(725, 776)
    mean_diameter = np.array([6.950051172684454, 6.892386122770274, 6.897441382470692, 6.996577106593007, 7.027228117561998, 6.898081464506564,
                              6.750612859271622, 6.529856445318642, 6.350915446862625, 6.548313187879266, 6.745710928895907, 6.747585938410225,
                              6.709534790118919, 6.684351436560332, 6.865052798725242, 7.032391941892341, 7.099514442575864, 7.163451376781643,
                              7.212522334091279, 7.199131566136183, 6.982740191070702, 6.83016921235479, 6.830767184877847, 6.849166027368458,
                              6.900199798132916, 6.948018494873731, 6.991244228723624, 7.0693146179652215, 7.1865852445224885, 7.153627949130994,
                              6.987134723118381, 6.766967836829394, 6.509229388346944, 6.601473000435695, 6.884615918290557, 7.0576068343728595,
                              7.184701083049932, 7.222166330815239, 6.2320534244538495, 6.211673054951369, 6.184806800490084, 7.155302901524694,
                              7.125447316625423, 7.05258391898559, 6.977032533049472, 7.350757182137674, 7.3532938946632145, 7.332989188149536,
                              7.238449869758395, 7.152761369850934, 7.077030040737611])
    d = {'Slice (I->S)': slices, 'VertLevel': vertlevel, 'MEAN(diameter_AP)': mean_diameter}
    df = pd.DataFrame(data=d)
    filename = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    df.to_csv(filename)
    return filename


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_compression_value_against_groundtruth():
    """Run the CLI script and verify that computed mscc value is equivalent to known ground truth value."""
    di, da, db = 6.85, 7.65, 7.02
    # FIXME: The results of "sct_compute_mscc" are not actually verified. Instead, the "mscc" function is called,
    #        and THOSE results are verified instead.
    # This was copied as-is from the existing 'sct_testing' test, but should be fixed at a later date.
    #sct_compute_compression.main(argv=['-di', str(di), '-da', str(da), '-db', str(db)])
    mscc = sct_compute_compression.metric_ratio(mi=di, ma=da, mb=db)
    assert mscc == pytest.approx(6.612133606, abs=1e-4)


def test_sct_compute_compression_check_missing_input_csv(tmp_path, dummy_3d_label_label, dummy_metrics_csv_pam50):
    """ Run sct_compute_compression when missing -i"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i-PAM50', dummy_metrics_csv_pam50, '-l', dummy_3d_label_label, '-o', filename])
        assert e.value.code == 2


def test_sct_compute_compression_check_missing_input_csv_pam50(tmp_path, dummy_3d_label_label, dummy_metrics_csv):
    """ Run sct_compute_compression when missing -i-PAM50"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_metrics_csv, '-l', dummy_3d_label_label, '-o', filename])
        assert e.value.code == 2


def test_sct_compute_compression_check_missing_input_l(tmp_path, dummy_metrics_csv, dummy_metrics_csv_pam50):
    """ Run sct_compute_mscc when missing -l"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_metrics_csv, '-i-PAM50', dummy_metrics_csv_pam50, '-o', filename])
        assert e.value.code == 2


def test_sct_compute_compression_check_wrong_sex(tmp_path, dummy_3d_label_label, dummy_metrics_csv, dummy_metrics_csv_pam50):
    """ Run sct_compute_compression when missing -l"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_metrics_csv, '-i-PAM50', dummy_metrics_csv_pam50, '-l', dummy_3d_label_label,
                                           '-sex', 'J', '-o', filename])
        assert e.value.code == 2


def test_sct_compute_compression_check_wrong_age(tmp_path, dummy_3d_label_label, dummy_metrics_csv, dummy_metrics_csv_pam50):
    """ Run sct_compute_compression when missing -l"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_metrics_csv, '-i-PAM50', dummy_metrics_csv_pam50, '-l', dummy_3d_label_label,
                                           '-age', '20', '-o', filename])
        assert e.value.code == 2


def test_sct_compute_compression_check_wrong_metric(tmp_path, dummy_3d_label_label, dummy_metrics_csv, dummy_metrics_csv_pam50):
    """ Run sct_compute_compression when missing -l"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_metrics_csv, '-i-PAM50', dummy_metrics_csv_pam50, '-l', dummy_3d_label_label,
                                           '-metric', 'MEAN', '20', '-o', filename])
        assert e.value.code == 2


def test_sct_compute_compression(tmp_path, dummy_3d_label_label, dummy_metrics_csv, dummy_metrics_csv_pam50):
    """ Run sct_compute_compression and chexk mscc and normalized mscc"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_compute_compression.main(argv=['-i', dummy_metrics_csv, '-i-PAM50', dummy_metrics_csv_pam50, '-l', dummy_3d_label_label,
                                       '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert row['Compression Level'] == '7.0'
        assert float(row['MEAN(diameter_AP) ratio']) == pytest.approx(11.471813876181447)
        assert float(row['Normalized MEAN(diameter_AP) ratio']) == pytest.approx(11.372089241980499)


def test_sct_compute_compression_sex_F(tmp_path, dummy_3d_label_label, dummy_metrics_csv, dummy_metrics_csv_pam50):
    """ Run sct_compute_compression and chexk mscc and normalized mscc"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_compute_compression.main(argv=['-i', dummy_metrics_csv, '-i-PAM50', dummy_metrics_csv_pam50, '-l', dummy_3d_label_label,
                                       '-sex', 'F', '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert row['Compression Level'] == '7.0'
        assert float(row['MEAN(diameter_AP) ratio']) == pytest.approx(11.471813876181447)
        assert float(row['Normalized MEAN(diameter_AP) ratio']) == pytest.approx(11.59367844245065)
