import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil

from unittest.mock import MagicMock, patch
import unittest
from step_runner import StepRunner
from hpc_driver import HPCDriver
from chemistry import Reaction, Molecule



class TestStepRunner(unittest.TestCase):

    def setUp(self):
        # Create mock objects
        self.mock_hpc_driver = MagicMock(spec=HPCDriver)
        self.mock_reaction = MagicMock(spec=Reaction)
        self.mock_molecule = MagicMock(spec=Molecule)

        # Initialize StepRunner with mock objects
        self.step_runner = StepRunner(
            hpc_driver=self.mock_hpc_driver, reaction=self.mock_reaction, molecule=self.mock_molecule)

    def test_initialization(self):
        # Test initialization
        self.assertEqual(self.step_runner.hpc_driver, self.mock_hpc_driver)
        self.assertEqual(self.step_runner.reaction, self.mock_reaction)
        self.assertEqual(self.step_runner.molecule, self.mock_molecule)
        self.assertEqual(self.step_runner.state, "INITIALISED")

    @patch('os.makedirs')
    @patch('shutil.rmtree')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_make_folder(self, mock_isdir, mock_exists, mock_rmtree, mock_makedirs):
        # Test make_folder method
        mock_exists.return_value = True
        mock_isdir.return_value = True

        self.step_runner.make_folder('test_folder')

        mock_rmtree.assert_called_once_with(
            os.path.join(os.getcwd(), 'test_folder'))
        mock_makedirs.assert_called_once_with(
            os.path.join(os.getcwd(), 'test_folder'))


    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="pattern: matched_line\n")
    @patch('subprocess.run')
    @patch('step_runner.StepRunner.shell_command')
    def test_grep_output(self, mock_shell_command, mock_subprocess_run, mock_open):
        # Setup mock for shell_command
        mock_shell_command.return_value.stdout = "matched_line\n"

        # Call the method
        result = self.step_runner.grep_output('pattern', 'test_file.txt')
        exit()
        # Assertions
        self.assertEqual(result, "matched_line")
    def test_geometry_optimisation(self, mock_shell_command, mock_scancel_job, mock_check_job_status, mock_submit_job, mock_grep_output, mock_make_folder):
        # Setup mocks
        mock_submit_job.return_value = 'job_id'
        mock_check_job_status.return_value = 'COMPLETED'
        mock_grep_output.return_value = 'HURRAY'
        # Call the method
        result = self.step_runner.geometry_optimisation()
        # Assertions
        self.assertTrue(result)
        self.assertEqual(self.step_runner.state, "OPT_COMPLETED")
        mock_make_folder.assert_called_once_with('OPT')
        mock_submit_job.assert_called()
        mock_check_job_status.assert_called()
        mock_grep_output.assert_called()
    @patch('step_runner.StepRunner.make_folder')
    @patch('step_runner.StepRunner.grep_output')
    @patch('step_runner.HPCDriver.submit_job')
    @patch('step_runner.HPCDriver.check_job_status')
    @patch('step_runner.HPCDriver.scancel_job')
    @patch('step_runner.HPCDriver.shell_command')
    def test_freq_job(self, mock_shell_command, mock_scancel_job, mock_check_job_status, mock_submit_job, mock_grep_output, mock_make_folder):
        # Setup mocks
        mock_submit_job.return_value = 'job_id'
        mock_check_job_status.return_value = 'COMPLETED'
        mock_grep_output.side_effect = [
            'VIBRATIONAL FREQUENCIES', '**imaginary mode*** -50.0 cm**-1']
        # Call the method
        result = self.step_runner.freq_job(self.mock_molecule, ts=True)
        # Assertions
        self.assertTrue(result)
        mock_submit_job.assert_called()
        mock_check_job_status.assert_called()
        mock_grep_output.assert_called()
    @patch('step_runner.StepRunner.make_folder')
    @patch('step_runner.StepRunner.grep_output')
    @patch('step_runner.HPCDriver.submit_job')
    @patch('step_runner.HPCDriver.check_job_status')
    @patch('step_runner.HPCDriver.scancel_job')
    @patch('step_runner.HPCDriver.shell_command')
    def test_neb_ts(self, mock_shell_command, mock_scancel_job, mock_check_job_status, mock_submit_job, mock_grep_output, mock_make_folder):
        # Setup mocks
        mock_submit_job.return_value = 'job_id'
        mock_check_job_status.return_value = 'COMPLETED'
        mock_grep_output.return_value = 'HURRAY'
        # Call the method
        result = self.step_runner.neb_ts()
        # Assertions
        self.assertTrue(result)
        mock_make_folder.assert_called_once_with('NEB')
        mock_submit_job.assert_called()
        mock_check_job_status.assert_called()
        mock_grep_output.assert_called()
    @patch('step_runner.StepRunner.make_folder')
    @patch('step_runner.StepRunner.grep_output')
    @patch('step_runner.HPCDriver.submit_job')
    @patch('step_runner.HPCDriver.check_job_status')
    @patch('step_runner.HPCDriver.scancel_job')
    @patch('step_runner.HPCDriver.shell_command')
    def test_neb_ci(self, mock_shell_command, mock_scancel_job, mock_check_job_status, mock_submit_job, mock_grep_output, mock_make_folder):
        # Setup mocks
        mock_submit_job.return_value = 'job_id'
        mock_check_job_status.return_value = 'COMPLETED'
        mock_grep_output.return_value = 'H U R R A Y'
        # Call the method
        result = self.step_runner.neb_ci()
        # Assertions
        self.assertTrue(result)
        mock_make_folder.assert_called_once_with('NEB_CI')
        mock_submit_job.assert_called()
        mock_check_job_status.assert_called()
        mock_grep_output.assert_called()
    @patch('step_runner.StepRunner.make_folder')
    @patch('step_runner.StepRunner.grep_output')
    @patch('step_runner.HPCDriver.submit_job')
    @patch('step_runner.HPCDriver.check_job_status')
    @patch('step_runner.HPCDriver.scancel_job')
    @patch('step_runner.HPCDriver.shell_command')
    def test_ts_opt(self, mock_shell_command, mock_scancel_job, mock_check_job_status, mock_submit_job, mock_grep_output, mock_make_folder):
        # Setup mocks
        mock_submit_job.return_value = 'job_id'
        mock_check_job_status.return_value = 'COMPLETED'
        mock_grep_output.return_value = 'HURRAY'
        # Call the method
        result = self.step_runner.ts_opt()
        # Assertions
        self.assertTrue(result)
        mock_make_folder.assert_called_once_with('TS')
        mock_submit_job.assert_called()
        mock_check_job_status.assert_called()
        mock_grep_output.assert_called()
    @patch('step_runner.StepRunner.make_folder')
    @patch('step_runner.StepRunner.grep_output')
    @patch('step_runner.HPCDriver.submit_job')
    @patch('step_runner.HPCDriver.check_job_status')
    @patch('step_runner.HPCDriver.scancel_job')
    @patch('step_runner.HPCDriver.shell_command')
    def test_irc_job(self, mock_shell_command, mock_scancel_job, mock_check_job_status, mock_submit_job, mock_grep_output, mock_make_folder):
        # Setup mocks
        mock_submit_job.return_value = 'job_id'
        mock_check_job_status.return_value = 'COMPLETED'
        mock_grep_output.return_value = 'HURRAY'
        # Call the method
        result = self.step_runner.irc_job()
        # Assertions
        self.assertTrue(result)
        mock_make_folder.assert_called_once_with('IRC')
        mock_submit_job.assert_called()
        mock_check_job_status.assert_called()
        mock_grep_output.assert_called()
    @patch('step_runner.StepRunner.make_folder')
    @patch('step_runner.StepRunner.grep_output')
    @patch('step_runner.HPCDriver.submit_job')
    @patch('step_runner.HPCDriver.check_job_status')
    @patch('step_runner.HPCDriver.scancel_job')
    @patch('step_runner.HPCDriver.shell_command')
    def test_sp_calc(self, mock_shell_command, mock_scancel_job, mock_check_job_status, mock_submit_job, mock_grep_output, mock_make_folder):
        # Setup mocks
        mock_submit_job.return_value = 'job_id'
        mock_check_job_status.return_value = 'COMPLETED'
        mock_grep_output.return_value = 'FINAL SINGLE POINT ENERGY'
        # Call the method
        result = self.step_runner.sp_calc()
        # Assertions
        self.assertTrue(result)
        mock_make_folder.assert_called_once_with('SP')
        mock_submit_job.assert_called()
        mock_check_job_status.assert_called()
        mock_grep_output.assert_called()


if __name__ == '__main__':
    unittest.main()
