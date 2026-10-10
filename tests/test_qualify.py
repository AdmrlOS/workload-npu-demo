import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class QualificationTests(unittest.TestCase):
    def run_fixture(self, verdict):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suite in ('rocket', 'ggml'):
                (root / suite).mkdir()
                (root / suite / 'CTestTestfile.cmake').touch()
            fake = root / 'ctest'
            fake.write_text('''#!/usr/bin/env python3
import os,pathlib,sys,xml.etree.ElementTree as ET
args=sys.argv; xml=pathlib.Path(args[args.index('--output-junit')+1])
names=['matmul_tiled_rocket','matmul_mt_rocket','conv2d_fp16_rocket'] if xml.stem=='rocket' else ['test-rocket-matmul','test-rocket-placement']
verdict=os.environ['FIXTURE_VERDICT']; tree=ET.Element('testsuite')
for name in names:
    if verdict=='missing' and name=='test-rocket-matmul': continue
    case=ET.SubElement(tree,'testcase',name=name,status='run',time='0.01')
    if verdict=='skip': ET.SubElement(case,'skipped')
    if verdict=='fail': ET.SubElement(case,'failure')
ET.ElementTree(tree).write(xml)
sys.exit(8 if verdict=='ctest_error' else 0)
''')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + ':' + os.environ['PATH'],
                       ROCKET_TEST_DIR=str(root / 'rocket'), GGML_TEST_DIR=str(root / 'ggml'),
                       FIXTURE_VERDICT=verdict)
            env.pop('HOST_KERNEL_LOG', None)
            env.pop('QUALIFY_FULL_SUITE', None)
            output = root / 'output'
            result = subprocess.run(['bash', str(ROOT / 'scripts/qualify.sh'), str(output)],
                                    env=env, capture_output=True, text=True, timeout=10)
            summary = json.loads((output / 'summary.json').read_text())
            return result.returncode, summary

    def test_required_hardware_passes_are_counted(self):
        status, summary = self.run_fixture('pass')
        self.assertEqual(status, 0)
        self.assertEqual(len(summary['hardware_passes']), 4)
        self.assertEqual(summary['host_kernel_review'], 'pending')
        self.assertIn('pending', summary['model_acceptance'])

    def test_skips_are_never_npu_success(self):
        status, summary = self.run_fixture('skip')
        self.assertEqual(status, 1)
        self.assertFalse(summary['primitive_gates_passed'])
        self.assertEqual(summary['hardware_passes'], [])

    def test_missing_failed_or_ctest_errors_reject(self):
        for verdict in ('missing', 'fail', 'ctest_error'):
            with self.subTest(verdict=verdict):
                status, summary = self.run_fixture(verdict)
                self.assertEqual(status, 1)
                self.assertFalse(summary['primitive_gates_passed'])
